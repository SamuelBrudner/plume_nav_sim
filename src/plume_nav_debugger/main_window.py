from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

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
        self._last_frame: np.ndarray | None = None
        self._last_event: object | None = None
        self._action_names: list[str] = []
        self._overlays_enabled: bool = True
        self._render_scheduled: bool = False
        self._seed: int | None = None
        self._start_xy: tuple[int, int] | None = None
        self._source_xy: tuple[int, int] | None = None
        self._goal_radius: float | None = None
        self._total_reward: float = 0.0
        self._last_rendered: QtGui.QPixmap | None = None

    @QtCore.Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray) -> None:
        # Expect HxWx3 uint8
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return
        h, w, c = frame.shape
        if c != 3:
            return
        if frame.dtype != np.uint8:
            return
        try:
            self._last_frame = np.asarray(frame).copy()
        except Exception:
            self._last_frame = frame
        self._schedule_render()

    @QtCore.Slot(object)
    def on_step_event(self, ev: object) -> None:
        self._last_event = ev
        try:
            info = getattr(ev, "info", None)
            if isinstance(info, dict) and "seed" in info and info["seed"] is not None:
                self._seed = int(info["seed"])
        except Exception:
            pass
        try:
            info = getattr(ev, "info", None)
            if isinstance(info, dict) and "total_reward" in info:
                self._total_reward = float(info["total_reward"])
            else:
                self._total_reward += float(getattr(ev, "reward", 0.0))
        except Exception:
            pass
        try:
            info = getattr(ev, "info", None)
            if isinstance(info, dict):
                src = self._coerce_xy(info.get("source_xy"))
                if src is not None:
                    self._source_xy = src
                if "goal_radius" in info and info["goal_radius"] is not None:
                    self._goal_radius = float(info["goal_radius"])
        except Exception:
            pass
        self._schedule_render()

    @QtCore.Slot(list)
    def on_action_names(self, names: list) -> None:
        try:
            self._action_names = [str(x) for x in (names or [])]
        except Exception:
            self._action_names = []
        self._schedule_render()

    @QtCore.Slot(int, object)
    def on_run_meta(self, seed_val: int, start_xy: object) -> None:
        try:
            self._seed = int(seed_val)
        except Exception:
            self._seed = None
        try:
            if isinstance(start_xy, tuple) and len(start_xy) == 2:
                self._start_xy = (int(start_xy[0]), int(start_xy[1]))
            else:
                self._start_xy = None
        except Exception:
            self._start_xy = None
        self._total_reward = 0.0
        self._last_event = None
        try:
            sender = self.sender()
            if sender is not None and hasattr(sender, "get_overlay_context"):
                ctx = sender.get_overlay_context()  # type: ignore[call-arg]
                if isinstance(ctx, dict):
                    src = self._coerce_xy(ctx.get("source_xy"))
                    if src is not None:
                        self._source_xy = src
                    if "goal_radius" in ctx and ctx["goal_radius"] is not None:
                        self._goal_radius = float(ctx["goal_radius"])
        except Exception:
            pass
        self._schedule_render()

    @QtCore.Slot(bool)
    def set_overlays_enabled(self, enabled: bool) -> None:
        self._overlays_enabled = bool(enabled)
        self._schedule_render()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._last_frame is not None:
            self._schedule_render()

    def _schedule_render(self) -> None:
        if self._render_scheduled:
            return
        self._render_scheduled = True
        QtCore.QTimer.singleShot(0, self._render_now)

    @QtCore.Slot()
    def _render_now(self) -> None:
        self._render_scheduled = False
        if self._last_frame is None:
            return

        frame = self._last_frame
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return
        h, w, c = frame.shape
        if c != 3:
            return

        img = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        base = QtGui.QPixmap.fromImage(img.copy())
        scaled = base.scaled(
            self.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )

        out = QtGui.QPixmap(scaled)
        if self._overlays_enabled:
            painter: QtGui.QPainter | None = None
            try:
                painter = QtGui.QPainter(out)
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                self._paint_overlays(
                    painter,
                    base_w=int(w),
                    base_h=int(h),
                    draw_w=int(out.width()),
                    draw_h=int(out.height()),
                )
            finally:
                try:
                    if painter is not None:
                        painter.end()
                except Exception:
                    pass

        self.setPixmap(out)
        self._last_rendered = out

    def _coerce_xy(self, value: object) -> tuple[int, int] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            try:
                if "x" in value and "y" in value:
                    return int(value["x"]), int(value["y"])
            except Exception:
                return None
        try:
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return int(value[0]), int(value[1])
        except Exception:
            return None
        try:
            x = getattr(value, "x", None)
            y = getattr(value, "y", None)
            if x is not None and y is not None:
                return int(x), int(y)
        except Exception:
            return None
        return None

    def _paint_overlays(
        self,
        painter: QtGui.QPainter,
        *,
        base_w: int,
        base_h: int,
        draw_w: int,
        draw_h: int,
    ) -> None:
        ev = self._last_event
        info = getattr(ev, "info", None) if ev is not None else None
        info_dict = info if isinstance(info, dict) else {}

        t = getattr(ev, "t", None)
        action = getattr(ev, "action", None)
        reward = getattr(ev, "reward", None)
        term = bool(getattr(ev, "terminated", False)) if ev is not None else False
        trunc = bool(getattr(ev, "truncated", False)) if ev is not None else False

        agent_xy = self._coerce_xy(
            info_dict.get("agent_xy") or info_dict.get("agent_position")
        )
        if agent_xy is None:
            agent_xy = self._start_xy

        source_xy = self._coerce_xy(info_dict.get("source_xy"))
        if source_xy is None:
            source_xy = self._coerce_xy(info_dict.get("goal_location"))
        if source_xy is None:
            source_xy = self._source_xy

        goal_radius = None
        try:
            if "goal_radius" in info_dict and info_dict["goal_radius"] is not None:
                goal_radius = float(info_dict["goal_radius"])
        except Exception:
            goal_radius = None
        if goal_radius is None:
            goal_radius = self._goal_radius

        heading_deg = None
        try:
            if (
                "agent_orientation" in info_dict
                and info_dict["agent_orientation"] is not None
            ):
                heading_deg = float(info_dict["agent_orientation"]) % 360.0
        except Exception:
            heading_deg = None

        action_idx = None
        try:
            if action is not None:
                action_idx = int(action)
        except Exception:
            action_idx = None
        action_name = None
        if (
            action_idx is not None
            and 0 <= action_idx < len(self._action_names)
            and self._action_names[action_idx]
        ):
            action_name = self._action_names[action_idx]

        total_reward = None
        try:
            if "total_reward" in info_dict and info_dict["total_reward"] is not None:
                total_reward = float(info_dict["total_reward"])
        except Exception:
            total_reward = None
        if total_reward is None:
            total_reward = float(self._total_reward)

        sx = (float(draw_w) / float(base_w)) if base_w > 0 else 1.0
        sy = (float(draw_h) / float(base_h)) if base_h > 0 else 1.0

        # Geometry overlays: source + goal radius
        if source_xy is not None:
            gx, gy = source_xy
            cx = (gx + 0.5) * sx
            cy = (gy + 0.5) * sy
            pen = QtGui.QPen(QtGui.QColor(255, 215, 0, 220), 2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(cx, cy), 6.0, 6.0)
            if goal_radius is not None and goal_radius > 0:
                rpx = float(goal_radius) * ((sx + sy) / 2.0)
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 215, 0, 120), 2))
                painter.drawEllipse(QtCore.QPointF(cx, cy), rpx, rpx)
            painter.setPen(QtGui.QColor(255, 215, 0, 220))
            painter.drawText(QtCore.QPointF(cx + 8.0, cy - 6.0), "GOAL")

        # Geometry overlays: agent + heading
        if agent_xy is not None:
            ax, ay = agent_xy
            cx = (ax + 0.5) * sx
            cy = (ay + 0.5) * sy
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 255, 220), 2))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(cx, cy), 6.0, 6.0)
            if heading_deg is not None:
                ang = math.radians(float(heading_deg))
                arrow_len = max(12.0, 3.0 * ((sx + sy) / 2.0))
                ex = cx + arrow_len * math.cos(ang)
                ey = cy + arrow_len * math.sin(ang)
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 255, 220), 2))
                painter.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(ex, ey))
                head_len = max(6.0, arrow_len * 0.35)
                left = ang + math.radians(150.0)
                right = ang - math.radians(150.0)
                painter.drawLine(
                    QtCore.QPointF(ex, ey),
                    QtCore.QPointF(
                        ex + head_len * math.cos(left), ey + head_len * math.sin(left)
                    ),
                )
                painter.drawLine(
                    QtCore.QPointF(ex, ey),
                    QtCore.QPointF(
                        ex + head_len * math.cos(right),
                        ey + head_len * math.sin(right),
                    ),
                )

        # HUD text (top-left)
        lines: list[str] = []
        if self._seed is not None:
            lines.append(f"seed={self._seed}")
        if t is not None:
            try:
                lines.append(f"t={int(t)}")
            except Exception:
                lines.append(f"t={t}")
        if action_idx is not None:
            if action_name:
                lines.append(f"action={action_idx} ({action_name})")
            else:
                lines.append(f"action={action_idx}")
        if reward is not None:
            try:
                lines.append(f"reward={float(reward):+.3f}")
            except Exception:
                lines.append(f"reward={reward}")
        lines.append(f"total_reward={float(total_reward):+.3f}")
        lines.append(f"terminated={term} truncated={trunc}")
        if agent_xy is not None:
            ax, ay = agent_xy
            if heading_deg is not None:
                lines.append(f"agent=({ax},{ay}) orient={heading_deg:.0f}° (post-step)")
            else:
                lines.append(f"agent=({ax},{ay})")
        if source_xy is not None:
            gx, gy = source_xy
            if goal_radius is not None:
                lines.append(f"goal=({gx},{gy}) r={goal_radius:g}")
            else:
                lines.append(f"goal=({gx},{gy})")

        text = "  ".join(lines)
        if not text:
            return
        fm = QtGui.QFontMetrics(painter.font())
        max_w = max(1, min(int(draw_w) - 16, 520))
        rect = fm.boundingRect(
            QtCore.QRect(0, 0, max_w, int(draw_h)),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap,
            text,
        )
        rect.moveTo(8, 8)
        bg = rect.adjusted(-6, -6, 6, 6)
        painter.fillRect(bg, QtGui.QColor(0, 0, 0, 160))
        painter.setPen(QtGui.QColor(255, 255, 255, 230))
        painter.drawText(
            rect,
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap,
            text,
        )


class ControlBar(QtWidgets.QWidget):
    start = QtCore.Signal()
    pause = QtCore.Signal()
    step = QtCore.Signal()
    step_back = QtCore.Signal()
    reset = QtCore.Signal(int)
    mode_changed = QtCore.Signal(str)
    load_replay = QtCore.Signal()
    seek_requested = QtCore.Signal(int)
    episode_seek_requested = QtCore.Signal(int)
    explore_toggled = QtCore.Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        main_row = QtWidgets.QHBoxLayout()

        self.start_btn = QtWidgets.QPushButton("Start")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.step_btn = QtWidgets.QPushButton("Step")
        self.step_back_btn = QtWidgets.QPushButton("Back")
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
        self.explore_check = QtWidgets.QCheckBox("Explore")
        self.seed_edit = QtWidgets.QLineEdit()
        self.seed_edit.setPlaceholderText("Seed")
        self.reset_btn = QtWidgets.QPushButton("Reset")

        main_row.addWidget(self.start_btn)
        main_row.addWidget(self.pause_btn)
        main_row.addWidget(self.step_btn)
        main_row.addWidget(self.step_back_btn)
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
        main_row.addWidget(self.explore_check)
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
        self.step_back_btn.clicked.connect(self.step_back)
        self.reset_btn.clicked.connect(self._emit_reset)
        self.mode_combo.currentTextChanged.connect(self.mode_changed)
        self.load_replay_btn.clicked.connect(self.load_replay)
        self.explore_check.toggled.connect(self.explore_toggled)
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
        self.explore_check.setEnabled(not is_replay)
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
        self.chk_overlays = QtWidgets.QCheckBox("Show frame overlays")
        self.chk_overlays.setChecked(self._prefs.show_overlays)
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
        layout.addRow(self.chk_overlays)
        layout.addRow("Default interval (ms)", self.spin_interval)
        layout.addRow("Theme", self.combo_theme)
        layout.addRow(btns)

    def get_prefs(self) -> DebuggerPreferences:
        self._prefs.show_pipeline = bool(self.chk_pipeline.isChecked())
        self._prefs.show_preview = bool(self.chk_preview.isChecked())
        self._prefs.show_sparkline = bool(self.chk_spark.isChecked())
        self._prefs.show_overlays = bool(self.chk_overlays.isChecked())
        self._prefs.default_interval_ms = int(self.spin_interval.value())
        self._prefs.theme = str(self.combo_theme.currentText())
        return self._prefs


class LiveConfigWidget(QtWidgets.QWidget):
    apply_requested = QtCore.Signal(object)  # DebuggerConfig

    def __init__(self, cfg: DebuggerConfig, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._updating = False
        self._applied = DebuggerConfig(**vars(cfg))
        self._draft = DebuggerConfig(**vars(cfg))
        self._presets = self._build_presets()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel("Preset:"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("Custom")
        for name in self._presets.keys():
            self.preset_combo.addItem(name)
        preset_row.addWidget(self.preset_combo, 1)
        layout.addLayout(preset_row)

        form = QtWidgets.QFormLayout()
        self.seed_edit = QtWidgets.QLineEdit()
        self.seed_edit.setPlaceholderText("e.g. 123 (blank = random)")
        self.max_steps_spin = QtWidgets.QSpinBox()
        self.max_steps_spin.setRange(1, 10_000_000)
        self.plume_combo = QtWidgets.QComboBox()
        self.plume_combo.addItems(["static", "movie"])
        self.action_combo = QtWidgets.QComboBox()
        self.action_combo.addItems(["oriented", "discrete", "run_tumble"])
        self.movie_dataset_edit = QtWidgets.QLineEdit()
        self.movie_dataset_edit.setPlaceholderText("registry id (optional)")
        self.movie_path_edit = QtWidgets.QLineEdit()
        self.movie_path_edit.setPlaceholderText("path to zarr/h5/avi (optional)")
        self.movie_browse_btn = QtWidgets.QToolButton()
        self.movie_browse_btn.setText("…")
        movie_path_row = QtWidgets.QWidget()
        movie_path_layout = QtWidgets.QHBoxLayout(movie_path_row)
        movie_path_layout.setContentsMargins(0, 0, 0, 0)
        movie_path_layout.addWidget(self.movie_path_edit, 1)
        movie_path_layout.addWidget(self.movie_browse_btn, 0)

        form.addRow("Seed", self.seed_edit)
        form.addRow("Plume", self.plume_combo)
        form.addRow("Action type", self.action_combo)
        form.addRow("Max steps", self.max_steps_spin)
        form.addRow("Movie dataset id", self.movie_dataset_edit)
        form.addRow("Movie path", movie_path_row)
        layout.addLayout(form)

        btn_row = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.revert_btn = QtWidgets.QPushButton("Revert")
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.revert_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addWidget(QtWidgets.QLabel("Resolved config"))
        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMinimumHeight(180)
        layout.addWidget(self.preview, 1)

        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        self.plume_combo.currentTextChanged.connect(self._on_plume_changed)
        self.action_combo.currentTextChanged.connect(self._on_fields_changed)
        self.max_steps_spin.valueChanged.connect(self._on_fields_changed)
        self.seed_edit.editingFinished.connect(self._on_fields_changed)
        self.movie_dataset_edit.editingFinished.connect(self._on_fields_changed)
        self.movie_path_edit.editingFinished.connect(self._on_fields_changed)
        self.movie_browse_btn.clicked.connect(self._browse_movie_path)
        self.apply_btn.clicked.connect(self._emit_apply)
        self.revert_btn.clicked.connect(self._revert_to_applied)

        self._sync_fields_from_config(self._draft)
        self._update_preview()

    def set_applied_config(self, cfg: DebuggerConfig) -> None:
        self._applied = DebuggerConfig(**vars(cfg))
        self._draft = DebuggerConfig(**vars(cfg))
        self.preset_combo.setCurrentText("Custom")
        self._sync_fields_from_config(self._draft)
        self._update_preview()

    def _build_presets(self) -> dict[str, DebuggerConfig]:
        repo_root = Path(__file__).resolve().parents[2]
        movie_demo = repo_root / "plug-and-play-demo" / "assets" / "demo_10s.zarr"
        gaussian_movie = (
            repo_root / "plug-and-play-demo" / "assets" / "gaussian_plume_demo.zarr"
        )
        presets: dict[str, DebuggerConfig] = {
            "Static quickstart": DebuggerConfig(
                plume="static",
                action_type="oriented",
                seed=123,
                max_steps=500,
            ),
            "Static small-grid deterministic": DebuggerConfig(
                plume="static",
                action_type="oriented",
                seed=0,
                max_steps=200,
                grid_size=(32, 32),
            ),
            "Movie demo (local zarr)": DebuggerConfig(
                plume="movie",
                action_type="run_tumble",
                seed=123,
                max_steps=500,
                movie_path=str(movie_demo),
            ),
            "Movie gaussian (local zarr)": DebuggerConfig(
                plume="movie",
                action_type="run_tumble",
                seed=123,
                max_steps=500,
                movie_path=str(gaussian_movie),
            ),
        }
        return presets

    def _sync_fields_from_config(self, cfg: DebuggerConfig) -> None:
        self._updating = True
        try:
            self.seed_edit.setText("" if cfg.seed is None else str(int(cfg.seed)))
        except Exception:
            self.seed_edit.setText("")
        try:
            self.max_steps_spin.setValue(max(1, int(cfg.max_steps)))
        except Exception:
            self.max_steps_spin.setValue(500)
        try:
            plume = str(cfg.plume or "static").strip().lower()
            if plume not in {"static", "movie"}:
                plume = "static"
            self.plume_combo.setCurrentText(plume)
        except Exception:
            self.plume_combo.setCurrentText("static")
        try:
            action = str(getattr(cfg, "action_type", "oriented") or "oriented")
            action = action.strip().lower()
            if action not in {"oriented", "discrete", "run_tumble"}:
                action = "oriented"
            self.action_combo.setCurrentText(action)
        except Exception:
            self.action_combo.setCurrentText("oriented")
        try:
            self.movie_dataset_edit.setText(
                "" if cfg.movie_dataset_id is None else str(cfg.movie_dataset_id)
            )
        except Exception:
            self.movie_dataset_edit.setText("")
        try:
            self.movie_path_edit.setText(
                "" if cfg.movie_path is None else str(cfg.movie_path)
            )
        except Exception:
            self.movie_path_edit.setText("")

        self._sync_movie_enabled()
        self._updating = False

    def _sync_movie_enabled(self) -> None:
        is_movie = str(self.plume_combo.currentText()).strip().lower() == "movie"
        self.movie_dataset_edit.setEnabled(is_movie)
        self.movie_path_edit.setEnabled(is_movie)
        self.movie_browse_btn.setEnabled(is_movie)

    @QtCore.Slot(str)
    def _on_preset_selected(self, name: str) -> None:
        if self._updating:
            return
        key = str(name).strip()
        if not key or key == "Custom":
            return
        preset = self._presets.get(key)
        if preset is None:
            return
        self._draft = DebuggerConfig(**vars(preset))
        self._sync_fields_from_config(self._draft)
        self._update_preview()

    @QtCore.Slot()
    def _revert_to_applied(self) -> None:
        self._draft = DebuggerConfig(**vars(self._applied))
        self.preset_combo.setCurrentText("Custom")
        self._sync_fields_from_config(self._draft)
        self._update_preview()

    @QtCore.Slot()
    def _on_fields_changed(self) -> None:
        if self._updating:
            return
        self._apply_fields_to_draft()
        self._update_preview()

    @QtCore.Slot(str)
    def _on_plume_changed(self, _text: str) -> None:
        if self._updating:
            return
        self._apply_fields_to_draft()
        self._sync_movie_enabled()
        # Enforce consistency: static plume clears movie selectors.
        plume = str(self.plume_combo.currentText()).strip().lower()
        if plume != "movie":
            self._draft.plume = "static"
            self._draft.movie_dataset_id = None
            self._draft.movie_path = None
            self._updating = True
            try:
                self.movie_dataset_edit.setText("")
                self.movie_path_edit.setText("")
            finally:
                self._updating = False
        self._update_preview()

    def _apply_fields_to_draft(self) -> None:
        plume = str(self.plume_combo.currentText()).strip().lower()
        if plume not in {"static", "movie"}:
            plume = "static"
        action = str(self.action_combo.currentText()).strip().lower()
        if action not in {"oriented", "discrete", "run_tumble"}:
            action = "oriented"

        seed_txt = self.seed_edit.text().strip()
        seed_val: Optional[int]
        if not seed_txt:
            seed_val = None
        elif seed_txt.lstrip("-").isdigit():
            seed_val = int(seed_txt)
        else:
            seed_val = self._draft.seed

        self._draft.plume = plume
        self._draft.action_type = action
        self._draft.max_steps = int(self.max_steps_spin.value())
        self._draft.seed = seed_val

        if plume == "movie":
            ds = self.movie_dataset_edit.text().strip()
            self._draft.movie_dataset_id = ds if ds else None
            mp = self.movie_path_edit.text().strip()
            self._draft.movie_path = mp if mp else None
        else:
            self._draft.movie_dataset_id = None
            self._draft.movie_path = None

    @QtCore.Slot()
    def _browse_movie_path(self) -> None:
        try:
            start_dir = str(Path.home())
            cur = self.movie_path_edit.text().strip()
            if cur:
                try:
                    start_dir = str(Path(cur).expanduser().resolve().parent)
                except Exception:
                    start_dir = str(Path.home())
            # Zarr is often a directory; allow either.
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select movie (zarr directory)", start_dir
            )
            if not path:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    "Select movie file",
                    start_dir,
                    "Movies (*.zarr *.h5 *.hdf5 *.avi *.mp4);;All files (*)",
                )
            if path:
                self.movie_path_edit.setText(str(path))
                if self.plume_combo.currentText().strip().lower() != "movie":
                    self.plume_combo.setCurrentText("movie")
                self._on_fields_changed()
        except Exception:
            pass

    @QtCore.Slot()
    def _emit_apply(self) -> None:
        self._apply_fields_to_draft()
        self.apply_requested.emit(DebuggerConfig(**vars(self._draft)))

    def _update_preview(self) -> None:
        try:
            payload = json.dumps(asdict(self._draft), indent=2, sort_keys=True)
            self.preview.setPlainText(payload)
        except Exception:
            self.preview.setPlainText("")


class ReplayConfigWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(QtWidgets.QLabel("Resolved replay config"))
        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMinimumHeight(220)
        layout.addWidget(self.preview, 1)
        self.set_payload({})

    def set_payload(self, payload: dict) -> None:
        try:
            txt = json.dumps(payload or {}, indent=2, sort_keys=True)
        except Exception:
            txt = ""
        self.preview.setPlainText(txt)


class ActionPanelWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QGridLayout(self)
        # Policy insight
        self.expected_action_label = QtWidgets.QLabel("action taken: -")
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
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
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
            action_label = self._act_model.state.action_label
            tie_label = None
            if self._act_model.state.distribution is not None:
                tie_label = self._distribution_tie(self._act_model.state.distribution)
            if tie_label:
                self.action_panel.expected_action_label.setText(
                    f"action taken: {action_label} (policy tie: {tie_label})"
                )
            else:
                self.action_panel.expected_action_label.setText(
                    f"action taken: {action_label}"
                )
            if self._act_model.state.distribution is not None:
                src = self._act_model.state.distribution_source or "probs"
                preview = self._format_distribution(self._act_model.state.distribution)
                self.action_panel.distribution_label.setText(
                    f"distribution ({src}): {preview}"
                )
            else:
                self.action_panel.distribution_label.setText("distribution: N/A")
        except Exception:
            logger.debug("InspectorWidget.on_step_event failed", exc_info=True)

    @QtCore.Slot(list)
    def on_action_names(self, names: list[str]) -> None:
        self._act_model.set_action_names(names)
        self.action_panel.on_action_names(names)
        self._update_strict_banner()

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
                    logger.debug("Pipeline extraction from mux failed", exc_info=True)
            # Update source label to indicate provider
            self.action_panel.source_label.setText("source: provider")
            self._update_strict_banner()
        except Exception:
            logger.debug("on_mux_changed failed", exc_info=True)
            self._mux = None
            self.action_panel.source_label.setText("source: none")
            self._update_strict_banner()

    def set_strict_provider_only(self, flag: bool) -> None:
        self._strict_provider_only = bool(flag)
        self._update_strict_banner()

    # Inspector display preferences passthrough
    def set_show_pipeline(self, flag: bool) -> None:
        try:
            self.obs_panel.set_show_pipeline(flag)
        except Exception:
            logger.debug("set_show_pipeline failed", exc_info=True)

    def set_show_preview(self, flag: bool) -> None:
        try:
            self.obs_panel.set_show_preview(flag)
        except Exception:
            logger.debug("set_show_preview failed", exc_info=True)

    def set_show_sparkline(self, flag: bool) -> None:
        try:
            self.obs_panel.set_show_sparkline(flag)
        except Exception:
            logger.debug("set_show_sparkline failed", exc_info=True)

    def _label_for_action_index(self, idx: int, *, include_index: bool = True) -> str:
        name = None
        try:
            if 0 <= idx < len(self._act_model.action_names):
                name = self._act_model.action_names[idx]
        except Exception:
            name = None
        if name:
            return f"{name}({idx})" if include_index else str(name)
        return str(idx)

    def _format_distribution(self, probs: list[float], *, max_items: int = 6) -> str:
        parts = []
        for idx, val in enumerate(probs):
            label = self._label_for_action_index(idx, include_index=True)
            parts.append(f"{label}={val:.2f}")
        if len(parts) > max_items:
            return ", ".join(parts[:max_items]) + ", ..."
        return ", ".join(parts)

    def _distribution_tie(self, probs: list[float], *, tol: float = 1e-3) -> str | None:
        if not probs:
            return None
        arr = np.asarray(probs, dtype=float).ravel()
        if arr.size == 0:
            return None
        max_val = float(np.max(arr))
        if not np.isfinite(max_val):
            return None
        top = [i for i, v in enumerate(arr) if abs(v - max_val) <= tol]
        if len(top) <= 1:
            return None
        labels = [self._label_for_action_index(i, include_index=False) for i in top]
        return "/".join(labels)

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
            if show and self.parent() is None and not self.isVisible():
                # Ensure the banner registers as visible for standalone widgets.
                self.show()
        except Exception:
            self.info_label.setVisible(False)
