from __future__ import annotations

from typing import Optional

try:
    from PySide6 import QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e

from plume_nav_debugger.config import DebuggerPreferences


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

