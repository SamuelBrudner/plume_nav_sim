from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

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
from plume_nav_debugger.replay_driver import ReplayDriver, ReplayLoadError
from plume_nav_debugger.widgets.action_panel_widget import ActionPanelWidget  # noqa: F401
from plume_nav_debugger.widgets.control_bar import ControlBar
from plume_nav_debugger.widgets.frame_view import FrameView
from plume_nav_debugger.widgets.inspector_widget import InspectorWidget
from plume_nav_debugger.widgets.live_config_widget import LiveConfigWidget
from plume_nav_debugger.widgets.observation_panel_widget import (  # noqa: F401
    ObservationPanelWidget,
)
from plume_nav_debugger.widgets.preferences_dialog import PreferencesDialog
from plume_nav_debugger.widgets.replay_config_widget import ReplayConfigWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Plume Nav Debugger (MVP)")
        self.resize(800, 700)

        cfg = DebuggerConfig.from_env()

        self.frame_view = FrameView()
        self.controls = ControlBar()
        self.inspector = InspectorWidget(self)
        self.live_config_widget = LiveConfigWidget(cfg)
        self.replay_config_widget = ReplayConfigWidget()
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

        # Dockable live config window (can float or dock)
        self.config_dock = QtWidgets.QDockWidget("Live Config", self)
        self.config_dock.setObjectName("LiveConfigDock")
        self.config_dock.setWidget(self.live_config_widget)
        self.config_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.config_dock)

        # Dockable replay config window (read-only)
        self.replay_config_dock = QtWidgets.QDockWidget("Replay Config", self)
        self.replay_config_dock.setObjectName("ReplayConfigDock")
        self.replay_config_dock.setWidget(self.replay_config_widget)
        self.replay_config_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.replay_config_dock)
        try:
            self.tabifyDockWidget(self.config_dock, self.replay_config_dock)
        except Exception:
            pass

        # Menu to toggle inspector visibility
        view_menu = self.menuBar().addMenu("View")
        self.action_toggle_inspector = QtGui.QAction("Inspector", self, checkable=True)
        self.action_toggle_inspector.setChecked(True)
        self.action_toggle_inspector.toggled.connect(self.inspector_dock.setVisible)
        self.inspector_dock.visibilityChanged.connect(
            self.action_toggle_inspector.setChecked
        )
        view_menu.addAction(self.action_toggle_inspector)
        self.action_toggle_live_config = QtGui.QAction(
            "Live Config", self, checkable=True
        )
        self.action_toggle_live_config.setChecked(True)
        self.action_toggle_live_config.toggled.connect(self.config_dock.setVisible)
        self.config_dock.visibilityChanged.connect(
            self.action_toggle_live_config.setChecked
        )
        view_menu.addAction(self.action_toggle_live_config)
        self.action_toggle_replay_config = QtGui.QAction(
            "Replay Config", self, checkable=True
        )
        self.action_toggle_replay_config.setChecked(True)
        self.action_toggle_replay_config.toggled.connect(
            self.replay_config_dock.setVisible
        )
        self.replay_config_dock.visibilityChanged.connect(
            self.action_toggle_replay_config.setChecked
        )
        view_menu.addAction(self.action_toggle_replay_config)
        self.action_toggle_overlays = QtGui.QAction(
            "Frame overlays", self, checkable=True
        )
        self.action_toggle_overlays.setChecked(True)
        self.action_toggle_overlays.toggled.connect(
            self.frame_view.set_overlays_enabled
        )
        view_menu.addAction(self.action_toggle_overlays)
        edit_menu = self.menuBar().addMenu("Edit")
        prefs_action = QtGui.QAction("Preferences…", self)
        prefs_action.triggered.connect(self._open_preferences)
        edit_menu.addAction(prefs_action)

        # Restore UI layout/state
        self._restore_ui_state()

        # Load preferences and configure driver
        self.prefs = DebuggerPreferences.initial_load()
        try:
            self.action_toggle_overlays.setChecked(bool(self.prefs.show_overlays))
        except Exception:
            logger.debug("Failed to restore overlay preference", exc_info=True)
        # Enforce strict provider-only mode via Inspector (always on; not configurable)
        self.live_driver = EnvDriver(cfg)
        self.replay_driver = ReplayDriver()
        self._active_driver = None
        self._active_mode = "live"
        self.live_config_widget.apply_requested.connect(self._on_live_config_apply)
        self.controls.start.connect(self._on_start_clicked)
        self.controls.pause.connect(self._on_pause_clicked)
        self.controls.step.connect(self._on_step_clicked)
        self.controls.step_back.connect(self._on_step_back_clicked)
        self.controls.reset.connect(self._on_reset_clicked)
        self.controls.interval_spin.valueChanged.connect(self._on_interval_changed)
        self.controls.policy_combo.currentIndexChanged.connect(self._on_policy_selected)
        self.controls.custom_load_btn.clicked.connect(self._on_custom_policy_load)
        self.controls.explore_toggled.connect(self._on_explore_toggled)
        self.controls.mode_changed.connect(self._on_mode_changed)
        self.controls.load_replay.connect(self._on_load_replay)
        self.controls.seek_requested.connect(self._on_seek_requested)
        self.controls.episode_seek_requested.connect(self._on_episode_seek_requested)
        try:
            self.live_driver.set_policy_explore(
                bool(self.controls.explore_check.isChecked())
            )
        except Exception:
            logger.debug("Failed to set initial explore state", exc_info=True)

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
        _signal_pairs = [
            ("frame_ready", self.frame_view.update_frame),
            ("step_done", self._on_step_event),
            ("step_done", self.inspector.on_step_event),
            ("step_done", self.frame_view.on_step_event),
            ("episode_finished", self._on_episode_finished),
            ("action_space_changed", self.inspector.on_action_names),
            ("action_space_changed", self.frame_view.on_action_names),
            ("policy_changed", self.inspector.on_policy_changed),
            ("provider_mux_changed", self.inspector.on_mux_changed),
            ("run_meta_changed", self._on_run_meta),
            ("run_meta_changed", self.frame_view.on_run_meta),
            ("error_occurred", self._on_driver_error),
            ("timeline_changed", self._on_timeline_changed),
        ]
        for signal_name, slot in _signal_pairs:
            sig = getattr(driver, signal_name, None)
            if sig is not None:
                try:
                    sig.connect(slot)
                except Exception:
                    logger.debug("Failed to connect %s on driver", signal_name, exc_info=True)

    def _disconnect_driver(self, driver: object) -> None:
        _signal_pairs = [
            ("frame_ready", self.frame_view.update_frame),
            ("step_done", self._on_step_event),
            ("step_done", self.inspector.on_step_event),
            ("step_done", self.frame_view.on_step_event),
            ("episode_finished", self._on_episode_finished),
            ("action_space_changed", self.inspector.on_action_names),
            ("action_space_changed", self.frame_view.on_action_names),
            ("policy_changed", self.inspector.on_policy_changed),
            ("provider_mux_changed", self.inspector.on_mux_changed),
            ("run_meta_changed", self._on_run_meta),
            ("run_meta_changed", self.frame_view.on_run_meta),
            ("error_occurred", self._on_driver_error),
            ("timeline_changed", self._on_timeline_changed),
        ]
        for signal_name, slot in _signal_pairs:
            sig = getattr(driver, signal_name, None)
            if sig is not None:
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass  # Not connected — expected for optional signals

    def _switch_driver(self, driver: object) -> None:
        if driver is self._active_driver:
            return
        if self._active_driver is not None:
            try:
                if hasattr(self._active_driver, "pause"):
                    self._active_driver.pause()  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Pause during driver switch failed", exc_info=True)
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
                logger.debug("Action names update failed during driver switch", exc_info=True)
        else:
            try:
                if self.replay_driver.is_loaded():
                    run_dir = getattr(self.replay_driver, "_run_dir", None)
                    label = run_dir.name if isinstance(run_dir, Path) else "loaded run"
                    self.controls.set_replay_label(label)
            except Exception:
                logger.debug("Replay label update failed", exc_info=True)
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
                logger.debug("Interval change failed", exc_info=True)

    @QtCore.Slot()
    def _on_pause_clicked(self) -> None:
        try:
            if self._active_driver is not None:
                self._active_driver.pause()  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Pause failed", exc_info=True)

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

    @QtCore.Slot()
    def _on_step_back_clicked(self) -> None:
        if self._active_mode == "replay":
            if not self.replay_driver.is_loaded():
                self.statusBar().showMessage(
                    "Load a replay run before stepping back", 2500
                )
                return
            try:
                self.replay_driver.step_back()
            except Exception as exc:
                self.statusBar().showMessage(f"Step back failed: {exc}", 3000)
            return
        try:
            self.live_driver.step_back()
        except Exception as exc:
            self.statusBar().showMessage(f"Step back failed: {exc}", 3000)

    def _setup_shortcuts(self) -> None:
        # Space: toggle start/pause
        space = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        space.activated.connect(self._toggle_run)

        # N: step once
        step = QtGui.QShortcut(QtGui.QKeySequence("N"), self)
        step.activated.connect(self._on_step_clicked)

        # B: step back
        back = QtGui.QShortcut(QtGui.QKeySequence("B"), self)
        back.activated.connect(self._on_step_back_clicked)

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
            logger.debug("Timeline refresh failed", exc_info=True)

    @QtCore.Slot(object)
    def _on_step_event(self, ev) -> None:
        try:
            info = getattr(ev, "info", None)
            if isinstance(info, dict) and "total_reward" in info:
                self._total_reward = float(info["total_reward"])
            else:
                self._total_reward += float(getattr(ev, "reward", 0.0))
        except Exception:
            logger.debug("Reward extraction failed in step event", exc_info=True)
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
        msg = str(message).strip() or "Unknown error"
        logger.warning("Driver error: %s", msg)
        self.statusBar().showMessage(msg, 6000)

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

    @QtCore.Slot(object)
    def _on_live_config_apply(self, cfg_obj: object) -> None:
        if not isinstance(cfg_obj, DebuggerConfig):
            return

        old_live = getattr(self, "live_driver", None)
        # Swap live driver (reconnect if in live mode); also close old env
        self.live_driver = EnvDriver(cfg_obj)
        try:
            self.live_driver.set_policy_explore(
                bool(self.controls.explore_check.isChecked())
            )
        except Exception:
            logger.debug("Failed to set explore on new driver", exc_info=True)
        try:
            self.live_config_widget.set_applied_config(cfg_obj)
        except Exception:
            logger.debug("Failed to update config widget", exc_info=True)

        if self._active_mode == "live":
            self._switch_driver(self.live_driver)
            QtCore.QTimer.singleShot(0, self.live_driver.initialize)
            QtCore.QTimer.singleShot(
                50,
                lambda: self.inspector.on_action_names(
                    self.live_driver.get_action_names()
                ),
            )
            QtCore.QTimer.singleShot(
                60,
                lambda: self.inspector.set_grid_size(*self.live_driver.get_grid_size()),
            )
            QtCore.QTimer.singleShot(
                80, lambda: self.inspector.set_strict_provider_only(True)
            )
        else:
            # Initialize in background so config errors surface immediately
            if hasattr(self.live_driver, "error_occurred"):
                try:
                    self.live_driver.error_occurred.connect(self._on_driver_error)  # type: ignore[attr-defined]
                except Exception:
                    logger.debug("Failed to connect error_occurred for background init", exc_info=True)
            try:
                self.live_driver.initialize()
            finally:
                if hasattr(self.live_driver, "error_occurred"):
                    try:
                        self.live_driver.error_occurred.disconnect(self._on_driver_error)  # type: ignore[attr-defined]
                    except (TypeError, RuntimeError):
                        pass  # Not connected

        try:
            if cfg_obj.seed is None:
                self.controls.seed_edit.setText("")
            else:
                self.controls.seed_edit.setText(str(int(cfg_obj.seed)))
        except Exception:
            logger.debug("Failed to update seed field", exc_info=True)

        try:
            if old_live is not None and hasattr(old_live, "close"):
                old_live.close()  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Old driver close failed", exc_info=True)
        try:
            if old_live is not None and hasattr(old_live, "deleteLater"):
                old_live.deleteLater()  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Old driver deleteLater failed", exc_info=True)

        self.statusBar().showMessage("Applied live config", 2000)

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
            logger.debug("Timeline change handler failed", exc_info=True)

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
            self.replay_config_widget.set_payload(
                self.replay_driver.get_resolved_replay_config()
            )
        except Exception:
            logger.debug("Failed to set replay config payload", exc_info=True)
        try:
            self.inspector.set_grid_size(*self.replay_driver.get_grid_size())
        except Exception:
            logger.debug("Failed to set grid size from replay", exc_info=True)
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
                logger.warning("Failed to save preferences to JSON file", exc_info=True)
            # Apply display-related prefs
            self._apply_inspector_prefs()
            self._apply_theme(self.prefs.theme)
            try:
                self.action_toggle_overlays.setChecked(bool(self.prefs.show_overlays))
            except Exception:
                logger.debug("Failed to update overlay toggle", exc_info=True)
            try:
                self.controls.interval_spin.setValue(
                    int(self.prefs.default_interval_ms)
                )
            except Exception:
                logger.debug("Failed to update interval spinner", exc_info=True)

    def _apply_inspector_prefs(self) -> None:
        try:
            self.inspector.set_show_pipeline(self.prefs.show_pipeline)
            self.inspector.set_show_preview(self.prefs.show_preview)
            self.inspector.set_show_sparkline(self.prefs.show_sparkline)
        except Exception:
            logger.debug("Inspector prefs application failed", exc_info=True)

    def _apply_theme(self, theme: str) -> None:
        try:
            if str(theme).lower() == "dark":
                self._enable_dark_palette()
            else:
                QtWidgets.QApplication.setPalette(
                    QtWidgets.QApplication.style().standardPalette()
                )
        except Exception:
            logger.debug("Theme application failed", exc_info=True)

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

    @QtCore.Slot(bool)
    def _on_explore_toggled(self, enabled: bool) -> None:
        if self._active_mode != "live":
            self.statusBar().showMessage("Explore is live-mode only", 2000)
            return
        try:
            self.live_driver.set_policy_explore(bool(enabled))
        except Exception as exc:
            logger.warning("Explore toggle failed: %s", exc)
            self.statusBar().showMessage(f"Explore toggle failed: {exc}", 3000)

    @QtCore.Slot()
    def _on_custom_policy_load(self) -> None:
        if self._active_mode != "live":
            self.statusBar().showMessage("Custom policies are live-mode only", 2500)
            return
        from plume_nav_sim.config.composition import load_policy

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
