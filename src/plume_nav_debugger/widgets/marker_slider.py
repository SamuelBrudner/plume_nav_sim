from __future__ import annotations

from collections.abc import Mapping, Sequence

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e


class MarkerSlider(QtWidgets.QSlider):
    """QSlider that can draw event markers (episode boundaries, terminations, etc.)."""

    def __init__(self, orientation: QtCore.Qt.Orientation, parent=None) -> None:
        super().__init__(orientation, parent)
        self._markers: dict[str, list[int]] = {}
        self._colors: dict[str, QtGui.QColor] = {
            "episode": QtGui.QColor(120, 120, 120, 180),
            "terminated": QtGui.QColor(46, 160, 67, 220),  # green
            "truncated": QtGui.QColor(230, 159, 0, 220),  # orange
            "goal": QtGui.QColor(46, 160, 67, 220),
        }
        self.rangeChanged.connect(lambda _a, _b: self.update())

    def set_markers(self, markers: Mapping[str, Sequence[int]] | None) -> None:
        """Set markers by kind.

        Expected indices are in the same domain as slider values (typically 0..max_idx).
        """

        if not markers:
            self._markers = {}
        else:
            out: dict[str, list[int]] = {}
            for kind, idxs in markers.items():
                if not isinstance(kind, str):
                    continue
                try:
                    cleaned = [int(x) for x in (idxs or [])]
                except Exception:
                    cleaned = []
                # Avoid duplicates for stable rendering.
                out[kind] = sorted(set(cleaned))
            self._markers = out
        self.update()

    def get_markers(self) -> dict[str, list[int]]:
        # Convenience for tests/debugging.
        return {k: list(v) for k, v in self._markers.items()}

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)

        if self.orientation() != QtCore.Qt.Horizontal:
            return
        if not self._markers:
            return

        vmin = int(self.minimum())
        vmax = int(self.maximum())
        if vmax <= vmin:
            return

        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QtWidgets.QStyle.CC_Slider,
            opt,
            QtWidgets.QStyle.SC_SliderGroove,
            self,
        )
        if not groove.isValid():
            return

        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
            # Draw subtle markers across the groove height.
            top = groove.top()
            bottom = groove.bottom()
            width = max(1, groove.width())

            def _x_for_value(value: int) -> int:
                frac = (float(value - vmin) / float(vmax - vmin)) if vmax != vmin else 0.0
                return int(groove.left() + frac * float(width))

            for kind, idxs in self._markers.items():
                if not idxs:
                    continue
                color = self._colors.get(kind, QtGui.QColor(200, 200, 200, 180))
                pen = QtGui.QPen(color, 1)
                painter.setPen(pen)
                for idx in idxs:
                    if idx < vmin or idx > vmax:
                        continue
                    x = _x_for_value(int(idx))
                    painter.drawLine(x, top, x, bottom)
        finally:
            painter.end()

