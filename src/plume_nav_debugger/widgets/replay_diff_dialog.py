from __future__ import annotations

import json
from typing import Any

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env."
    ) from e


class ReplayDiffDialog(QtWidgets.QDialog):
    """Non-modal dialog for presenting replay divergence diffs."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Replay Divergence")
        self.setModal(False)
        self.resize(820, 560)

        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setWordWrap(True)

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        try:
            fixed = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            self.text.setFont(fixed)
        except Exception:
            pass

        self.copy_btn = QtWidgets.QPushButton("Copy")
        self.close_btn = QtWidgets.QPushButton("Close")
        self.copy_btn.clicked.connect(self._copy_to_clipboard)
        self.close_btn.clicked.connect(self.close)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.copy_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.close_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.summary_label, stretch=0)
        layout.addWidget(self.text, stretch=1)
        layout.addLayout(btn_row)

        self._payload: dict[str, Any] | None = None

    def set_payload(self, payload: object, *, message: str | None = None) -> None:
        data = self._normalize_payload(payload)
        self._payload = data
        self.summary_label.setText(message or self._format_summary(data))
        self.text.setPlainText(self._format_body(data))

    def _normalize_payload(self, payload: object) -> dict[str, Any]:
        if isinstance(payload, dict):
            return dict(payload)
        if hasattr(payload, "to_dict"):
            try:
                obj = payload.to_dict()  # type: ignore[attr-defined]
                if isinstance(obj, dict):
                    return obj
                return {"payload": obj}
            except Exception:
                return {"payload": str(payload)}
        return {"payload": payload}

    def _format_summary(self, data: dict[str, Any]) -> str:
        idx = data.get("global_step_index")
        ep = data.get("episode_id")
        t = data.get("episode_step")
        fields: list[str] = []
        mismatches = data.get("mismatches")
        if isinstance(mismatches, list):
            for m in mismatches:
                if isinstance(m, dict) and isinstance(m.get("field"), str):
                    fields.append(str(m["field"]))
        parts: list[str] = []
        if idx is not None:
            parts.append(f"step={idx}")
        if ep is not None:
            parts.append(f"episode={ep}")
        if t is not None:
            parts.append(f"t={t}")
        if parts:
            headline = "Replay divergence detected (" + ", ".join(parts) + ")"
        else:
            headline = "Replay divergence detected"
        if fields:
            return headline + "\nMismatched fields: " + ", ".join(fields)
        return headline

    def _format_body(self, data: dict[str, Any]) -> str:
        try:
            return json.dumps(data, indent=2, sort_keys=False, default=str)
        except Exception:
            return str(data)

    @QtCore.Slot()
    def _copy_to_clipboard(self) -> None:
        try:
            QtWidgets.QApplication.clipboard().setText(self.text.toPlainText())
        except Exception:
            pass


__all__ = ["ReplayDiffDialog"]

