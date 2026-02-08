from __future__ import annotations

import json
from typing import Optional

try:
    from PySide6 import QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e


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

