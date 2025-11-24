from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e

from plume_nav_debugger.config import DebuggerPreferences
from plume_nav_debugger.env_driver import DebuggerConfig, EnvDriver
from plume_nav_debugger.inspector.introspection import format_pipeline
from plume_nav_debugger.inspector.models import ActionPanelModel, ObservationPanelModel
from plume_nav_debugger.inspector.plots import normalize_series_to_polyline
from plume_nav_debugger.odc.mux import ProviderMux


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

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.step_btn = QtWidgets.QPushButton("Step")
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

        layout.addWidget(self.start_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.step_btn)
        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Policy:"))
        layout.addWidget(self.policy_combo)
        layout.addWidget(self.custom_policy_edit)
        layout.addWidget(self.custom_load_btn)
        layout.addSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Interval (ms):"))
        layout.addWidget(self.interval_spin)
        layout.addStretch(1)
        layout.addWidget(QtWidgets.QLabel("Seed:"))
        layout.addWidget(self.seed_edit)
        layout.addWidget(self.reset_btn)

        self.start_btn.clicked.connect(self.start)
        self.pause_btn.clicked.connect(self.pause)
        self.step_btn.clicked.connect(self.step)
        self.reset_btn.clicked.connect(self._emit_reset)
        # Expose additional signals
        # Note: MainWindow will connect signals directly to handlers

    @QtCore.Slot()
    def _emit_reset(self) -> None:
        text = self.seed_edit.text().strip()
        # If seed field is empty or not an int, emit a sentinel -1 and let driver reuse last seed
        if text.isdigit():
            self.reset.emit(int(text))
        else:
            # Use -1 to indicate "reuse last episode seed"
            self.reset.emit(-1)


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
        cfg = DebuggerConfig()
        # Enforce strict provider-only mode via Inspector (always on; not configurable)
        self.driver = EnvDriver(cfg)
        self.driver.frame_ready.connect(self.frame_view.update_frame)
        self.driver.step_done.connect(self._on_step_event)
        self.driver.episode_finished.connect(self._on_episode_finished)
        self.driver.step_done.connect(self.inspector.on_step_event)
        self.driver.action_space_changed.connect(self.inspector.on_action_names)
        self.driver.policy_changed.connect(self.inspector.on_policy_changed)
        self.driver.provider_mux_changed.connect(self.inspector.on_mux_changed)
        self.driver.run_meta_changed.connect(self._on_run_meta)
        self.controls.start.connect(self._on_start_clicked)
        self.controls.pause.connect(self.driver.pause)
        self.controls.step.connect(self.driver.step_once)
        self.controls.reset.connect(self._on_reset_clicked)
        self.controls.interval_spin.valueChanged.connect(self._on_interval_changed)
        self.controls.policy_combo.currentIndexChanged.connect(self._on_policy_selected)
        self.controls.custom_load_btn.clicked.connect(self._on_custom_policy_load)

        # Status bar showing step/total reward/flags and run meta
        self._status = QtWidgets.QLabel("ready")
        self._run_meta = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self._status, 1)
        self.statusBar().addPermanentWidget(self._run_meta, 0)

        # Keyboard shortcuts
        self._setup_shortcuts()

        # Initialize once to show first frame
        QtCore.QTimer.singleShot(0, self.driver.initialize)
        # Also refresh action names shortly after init
        QtCore.QTimer.singleShot(
            50, lambda: self.inspector.on_action_names(self.driver.get_action_names())
        )
        # Initialize start override ranges based on grid size
        QtCore.QTimer.singleShot(
            60, lambda: self.inspector.set_grid_size(*self.driver.get_grid_size())
        )
        QtCore.QTimer.singleShot(
            70,
            lambda: self.inspector.set_observation_pipeline_from_env(
                getattr(self.driver, "_env", None)
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

            settings = _QtCore.QSettings("plume-nav-sim", "Debugger")
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("windowState", self.saveState())
        except Exception:
            pass
        super().closeEvent(event)

    def _restore_ui_state(self) -> None:
        try:
            from PySide6 import QtCore as _QtCore

            settings = _QtCore.QSettings("plume-nav-sim", "Debugger")
            geom = settings.value("geometry")
            if geom is not None:
                self.restoreGeometry(geom)
            st = settings.value("windowState")
            if st is not None:
                self.restoreState(st)
        except Exception:
            pass

    # UI wiring --------------------------------------------------------------
    @QtCore.Slot()
    def _on_start_clicked(self) -> None:
        interval = int(self.controls.interval_spin.value())
        self.driver.start(interval)

    @QtCore.Slot(int)
    def _on_interval_changed(self, value: int) -> None:
        if self.driver.is_running():
            self.driver.start(int(value))

    def _setup_shortcuts(self) -> None:
        # Space: toggle start/pause
        space = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        space.activated.connect(self._toggle_run)

        # N: step once
        step = QtGui.QShortcut(QtGui.QKeySequence("N"), self)
        step.activated.connect(self.driver.step_once)

        # R: reset with seed from edit (delegates to existing handler)
        reset = QtGui.QShortcut(QtGui.QKeySequence("R"), self)
        reset.activated.connect(self.controls._emit_reset)  # type: ignore[attr-defined]

    @QtCore.Slot()
    def _toggle_run(self) -> None:
        if self.driver.is_running():
            self.driver.pause()
        else:
            self._on_start_clicked()

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
        # Interpret -1 as "reuse last episode seed"
        self.driver.reset(None if seed == -1 else seed)

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
        # Map combo index to built-in policies
        label = self.controls.policy_combo.itemText(idx)
        try:
            if label == "Greedy TD (bacterial)":
                from plume_nav_sim.policies import TemporalDerivativePolicy

                policy = TemporalDerivativePolicy(
                    eps=0.0, eps_after_turn=0.0, uniform_random_on_non_increase=True
                )
                self.driver.set_policy(policy, seed=self._current_seed_value())
                self.statusBar().showMessage(
                    "Loaded Greedy TD (bacterial) policy", 1500
                )
            elif label == "Deterministic TD":
                from plume_nav_sim.policies import TemporalDerivativeDeterministicPolicy

                policy = TemporalDerivativeDeterministicPolicy()
                self.driver.set_policy(policy, seed=self._current_seed_value())
                self.statusBar().showMessage("Loaded Deterministic TD policy", 1500)
            elif label == "Stochastic TD":
                from plume_nav_sim.policies import TemporalDerivativePolicy

                policy = TemporalDerivativePolicy(eps=0.05, eps_after_turn=0.0)
                self.driver.set_policy(policy, seed=self._current_seed_value())
                self.statusBar().showMessage("Loaded Stochastic TD policy", 1500)
            elif label == "Random Sampler":
                # Use the driver's default random sampler
                policy = (
                    self.driver._make_default_policy()
                )  # noqa: SLF001 - internal is fine for UI wiring
                self.driver.set_policy(policy, seed=self._current_seed_value())
                self.statusBar().showMessage("Loaded Random Sampler policy", 1500)
        except Exception as e:  # pragma: no cover - UI safety
            self.statusBar().showMessage(f"Policy load failed: {e}", 3000)

    @QtCore.Slot()
    def _on_custom_policy_load(self) -> None:
        from plume_nav_sim.compose.policy_loader import load_policy

        spec = self.controls.custom_policy_edit.text().strip()
        if not spec:
            self.statusBar().showMessage("Enter custom policy as module:Attr", 2500)
            return
        try:
            loaded = load_policy(spec)
            self.driver.set_policy(loaded.obj, seed=self._current_seed_value())
            self.statusBar().showMessage(f"Loaded custom policy: {loaded.spec}", 2000)
            # Names may change with new policy/env combination
            self.inspector.on_action_names(self.driver.get_action_names())
        except Exception as e:  # pragma: no cover - UI safety
            self.statusBar().showMessage(f"Custom policy load failed: {e}", 4000)

    def _current_seed_value(self) -> Optional[int]:
        text = self.controls.seed_edit.text().strip()
        return int(text) if text.isdigit() else self.driver.config.seed

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
