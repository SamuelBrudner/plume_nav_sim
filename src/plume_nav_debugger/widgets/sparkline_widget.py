from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e

from plume_nav_debugger.inspector.plots import normalize_series_to_polyline


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

