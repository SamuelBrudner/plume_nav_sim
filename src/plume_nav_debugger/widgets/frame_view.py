from __future__ import annotations

import math

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e


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

