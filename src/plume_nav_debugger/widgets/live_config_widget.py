from __future__ import annotations

import contextlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

try:
    from PySide6 import QtCore, QtWidgets
except Exception as e:  # pragma: no cover - GUI import guard
    raise RuntimeError(
        "PySide6 is required for the debugger app. Install it in your env (e.g., conda run -n plume-nav-sim pip install PySide6)."
    ) from e

from plume_nav_debugger.env_driver import DebuggerConfig


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
        self.action_names_edit = QtWidgets.QLineEdit()
        self.action_names_edit.setPlaceholderText("comma-separated (optional)")
        self.movie_dataset_edit = QtWidgets.QLineEdit()
        self.movie_dataset_edit.setPlaceholderText("registry id (optional)")
        self.movie_path_edit = QtWidgets.QLineEdit()
        self.movie_path_edit.setPlaceholderText("path to zarr/h5/avi (optional)")
        self.movie_browse_btn = QtWidgets.QToolButton()
        self.movie_browse_btn.setText("…")
        self.movie_auto_download_check = QtWidgets.QCheckBox()
        self.movie_auto_download_check.setText("Download if missing (registry datasets only)")
        self.movie_cache_root_edit = QtWidgets.QLineEdit()
        self.movie_cache_root_edit.setPlaceholderText(
            "override cache root (registry datasets only)"
        )
        self.movie_cache_root_browse_btn = QtWidgets.QToolButton()
        self.movie_cache_root_browse_btn.setText("…")
        movie_path_row = QtWidgets.QWidget()
        movie_path_layout = QtWidgets.QHBoxLayout(movie_path_row)
        movie_path_layout.setContentsMargins(0, 0, 0, 0)
        movie_path_layout.addWidget(self.movie_path_edit, 1)
        movie_path_layout.addWidget(self.movie_browse_btn, 0)

        movie_cache_root_row = QtWidgets.QWidget()
        movie_cache_root_layout = QtWidgets.QHBoxLayout(movie_cache_root_row)
        movie_cache_root_layout.setContentsMargins(0, 0, 0, 0)
        movie_cache_root_layout.addWidget(self.movie_cache_root_edit, 1)
        movie_cache_root_layout.addWidget(self.movie_cache_root_browse_btn, 0)

        form.addRow("Seed", self.seed_edit)
        form.addRow("Plume", self.plume_combo)
        form.addRow("Action type", self.action_combo)
        form.addRow("Action names", self.action_names_edit)
        form.addRow("Max steps", self.max_steps_spin)
        form.addRow("Movie dataset id", self.movie_dataset_edit)
        form.addRow("Movie path", movie_path_row)
        form.addRow("Movie auto-download", self.movie_auto_download_check)
        form.addRow("Movie cache root", movie_cache_root_row)
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
        self.action_names_edit.editingFinished.connect(self._on_fields_changed)
        self.max_steps_spin.valueChanged.connect(self._on_fields_changed)
        self.seed_edit.editingFinished.connect(self._on_fields_changed)
        self.movie_dataset_edit.editingFinished.connect(self._on_fields_changed)
        self.movie_path_edit.editingFinished.connect(self._on_fields_changed)
        self.movie_auto_download_check.stateChanged.connect(
            lambda _v=None: self._on_fields_changed()
        )
        self.movie_cache_root_edit.editingFinished.connect(self._on_fields_changed)
        self.movie_browse_btn.clicked.connect(self._browse_movie_path)
        self.movie_cache_root_browse_btn.clicked.connect(self._browse_movie_cache_root)
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
        # This file lives at src/plume_nav_debugger/widgets/*; repo root is 3 parents up.
        repo_root = Path(__file__).resolve().parents[3]
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
            names = getattr(cfg, "action_names_override", None)
            if isinstance(names, list):
                self.action_names_edit.setText(", ".join(str(x) for x in names))
            else:
                self.action_names_edit.setText("")
        except Exception:
            self.action_names_edit.setText("")
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
        try:
            self.movie_auto_download_check.setChecked(bool(cfg.movie_auto_download))
        except Exception:
            self.movie_auto_download_check.setChecked(False)
        try:
            self.movie_cache_root_edit.setText(
                "" if cfg.movie_cache_root is None else str(cfg.movie_cache_root)
            )
        except Exception:
            self.movie_cache_root_edit.setText("")

        self._sync_movie_enabled()
        self._updating = False

    def _sync_movie_enabled(self) -> None:
        is_movie = str(self.plume_combo.currentText()).strip().lower() == "movie"
        has_dataset_id = bool(self.movie_dataset_edit.text().strip())
        self.movie_dataset_edit.setEnabled(is_movie)
        self.movie_path_edit.setEnabled(is_movie)
        self.movie_browse_btn.setEnabled(is_movie)
        self.movie_auto_download_check.setEnabled(is_movie and has_dataset_id)
        self.movie_cache_root_edit.setEnabled(is_movie and has_dataset_id)
        self.movie_cache_root_browse_btn.setEnabled(is_movie and has_dataset_id)

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
        self._sync_movie_enabled()
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
            self._draft.movie_auto_download = False
            self._draft.movie_cache_root = None
            self._updating = True
            try:
                self.movie_dataset_edit.setText("")
                self.movie_path_edit.setText("")
                self.movie_auto_download_check.setChecked(False)
                self.movie_cache_root_edit.setText("")
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
        names_txt = self.action_names_edit.text().strip()
        if names_txt:
            parsed_names = [part.strip() for part in names_txt.split(",")]
            parsed_names = [name for name in parsed_names if name]
            self._draft.action_names_override = parsed_names or None
        else:
            self._draft.action_names_override = None

        if plume == "movie":
            ds = self.movie_dataset_edit.text().strip()
            self._draft.movie_dataset_id = ds if ds else None
            mp = self.movie_path_edit.text().strip()
            self._draft.movie_path = mp if mp else None
            self._draft.movie_auto_download = bool(
                self.movie_auto_download_check.isChecked()
            )
            cr = self.movie_cache_root_edit.text().strip()
            self._draft.movie_cache_root = cr if cr else None
        else:
            self._draft.movie_dataset_id = None
            self._draft.movie_path = None
            self._draft.movie_auto_download = False
            self._draft.movie_cache_root = None

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
    def _browse_movie_cache_root(self) -> None:
        try:
            start_dir = str(Path.home())
            cur = self.movie_cache_root_edit.text().strip()
            if cur:
                with contextlib.suppress(Exception):
                    start_dir = str(Path(cur).expanduser().resolve())
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select cache root (registry datasets)", start_dir
            )
            if path:
                self.movie_cache_root_edit.setText(str(path))
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

