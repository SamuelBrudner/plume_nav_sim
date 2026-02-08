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


class _DatasetDownloadWorker(QtCore.QObject):
    finished = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, *, dataset_id: str, cache_root: Optional[str]) -> None:
        super().__init__()
        self._dataset_id = str(dataset_id)
        self._cache_root = cache_root

    @QtCore.Slot()
    def run(self) -> None:
        try:
            from plume_nav_sim.data_zoo.download import ensure_dataset_available

            cache_root_path = (
                None
                if self._cache_root is None or not str(self._cache_root).strip()
                else Path(str(self._cache_root)).expanduser()
            )
            path = ensure_dataset_available(
                self._dataset_id,
                cache_root=cache_root_path,
                auto_download=True,
            )
            self.finished.emit(str(path))
        except Exception as exc:
            self.failed.emit(str(exc))


class LiveConfigWidget(QtWidgets.QWidget):
    apply_requested = QtCore.Signal(object)  # DebuggerConfig

    def __init__(self, cfg: DebuggerConfig, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._updating = False
        self._movie_download_thread: QtCore.QThread | None = None
        self._movie_download_worker: _DatasetDownloadWorker | None = None
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

        self.movie_dataset_combo = QtWidgets.QComboBox()
        self.movie_dataset_combo.addItem("<none>", userData=None)
        self._registry_ids: list[str] = []
        try:
            from plume_nav_sim.data_zoo.registry import get_dataset_registry

            self._registry_ids = sorted(get_dataset_registry().keys())
        except Exception:
            self._registry_ids = []
        for ds_id in self._registry_ids:
            self.movie_dataset_combo.addItem(ds_id, userData=ds_id)

        self.movie_registry_status_label = QtWidgets.QLabel("")
        self.movie_registry_path_edit = QtWidgets.QLineEdit()
        self.movie_registry_path_edit.setReadOnly(True)
        self.movie_registry_path_edit.setPlaceholderText("registry cache path")
        self.movie_download_btn = QtWidgets.QPushButton("Download")
        self.movie_download_btn.clicked.connect(self._download_selected_dataset)
        self.movie_download_progress = QtWidgets.QProgressBar()
        self.movie_download_progress.setRange(0, 0)  # indeterminate
        self.movie_download_progress.setVisible(False)

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

        movie_registry_status_row = QtWidgets.QWidget()
        movie_registry_status_layout = QtWidgets.QHBoxLayout(movie_registry_status_row)
        movie_registry_status_layout.setContentsMargins(0, 0, 0, 0)
        movie_registry_status_layout.addWidget(self.movie_registry_status_label, 1)
        movie_registry_status_layout.addWidget(self.movie_download_btn, 0)
        movie_registry_status_layout.addWidget(self.movie_download_progress, 0)

        form.addRow("Seed", self.seed_edit)
        form.addRow("Plume", self.plume_combo)
        form.addRow("Action type", self.action_combo)
        form.addRow("Action names", self.action_names_edit)
        form.addRow("Max steps", self.max_steps_spin)
        form.addRow("Movie dataset (registry)", self.movie_dataset_combo)
        form.addRow("Movie dataset status", movie_registry_status_row)
        form.addRow("Movie dataset path", self.movie_registry_path_edit)
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
        self.movie_dataset_combo.currentIndexChanged.connect(self._on_fields_changed)
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
            self._set_selected_dataset_id(
                None if cfg.movie_dataset_id is None else str(cfg.movie_dataset_id)
            )
        except Exception:
            self._set_selected_dataset_id(None)
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
        self._update_movie_registry_status()
        self._updating = False

    def _sync_movie_enabled(self) -> None:
        is_movie = str(self.plume_combo.currentText()).strip().lower() == "movie"
        has_dataset_id = bool(self._get_selected_dataset_id())
        self.movie_dataset_combo.setEnabled(is_movie)
        self.movie_registry_status_label.setEnabled(is_movie)
        self.movie_registry_path_edit.setEnabled(is_movie)
        self.movie_download_btn.setEnabled(is_movie and has_dataset_id)
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
        self._update_movie_registry_status()
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
                self._set_selected_dataset_id(None)
                self.movie_path_edit.setText("")
                self.movie_auto_download_check.setChecked(False)
                self.movie_cache_root_edit.setText("")
            finally:
                self._updating = False
            self._update_movie_registry_status()
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
            self._draft.movie_dataset_id = self._get_selected_dataset_id()
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

    def _get_selected_dataset_id(self) -> Optional[str]:
        try:
            ds = self.movie_dataset_combo.currentData()
            if isinstance(ds, str):
                ds = ds.strip()
                return ds or None
        except Exception:
            pass
        try:
            txt = str(self.movie_dataset_combo.currentText()).strip()
            if txt == "<none>":
                return None
            return txt or None
        except Exception:
            return None

    def _set_selected_dataset_id(self, dataset_id: Optional[str]) -> None:
        ds = None if dataset_id is None else str(dataset_id).strip()
        prev_updating = self._updating
        self._updating = True
        try:
            if not ds:
                self.movie_dataset_combo.setCurrentIndex(0)
                return
            idx = self.movie_dataset_combo.findData(ds)
            if idx < 0:
                # Preserve unknown ids by adding them to the picker (still editable via config).
                self.movie_dataset_combo.addItem(f"{ds}", userData=ds)
                idx = self.movie_dataset_combo.findData(ds)
            if idx >= 0:
                self.movie_dataset_combo.setCurrentIndex(idx)
        finally:
            self._updating = prev_updating

    def _resolve_registry_expected_path(self, dataset_id: str) -> Optional[Path]:
        try:
            from plume_nav_sim.data_zoo.registry import DEFAULT_CACHE_ROOT, describe_dataset
        except Exception:
            return None

        try:
            entry = describe_dataset(dataset_id)
        except Exception:
            return None

        cache_root_txt = self.movie_cache_root_edit.text().strip()
        cache_root = (
            DEFAULT_CACHE_ROOT
            if not cache_root_txt
            else Path(cache_root_txt).expanduser()
        )
        cache_dir = entry.cache_path(cache_root)
        return cache_dir / entry.expected_root

    def _update_movie_registry_status(self) -> None:
        is_movie = str(self.plume_combo.currentText()).strip().lower() == "movie"
        ds = self._get_selected_dataset_id()
        if not is_movie or not ds:
            self.movie_registry_status_label.setText("")
            self.movie_registry_path_edit.setText("")
            self.movie_download_btn.setEnabled(False)
            return

        expected = self._resolve_registry_expected_path(ds)
        if expected is None:
            self.movie_registry_status_label.setText("Registry unavailable or unknown dataset id")
            self.movie_registry_path_edit.setText("")
            self.movie_download_btn.setEnabled(False)
            return

        self.movie_registry_path_edit.setText(str(expected))
        cached = expected.exists()
        self.movie_registry_status_label.setText("Cached" if cached else "Not cached")
        self.movie_download_btn.setEnabled(not cached)

    @QtCore.Slot()
    def _download_selected_dataset(self) -> None:
        ds = self._get_selected_dataset_id()
        if not ds:
            return

        if (
            self._movie_download_thread is not None
            and self._movie_download_thread.isRunning()
        ):
            return  # One download at a time

        self.movie_download_btn.setEnabled(False)
        self.movie_download_progress.setVisible(True)
        self.movie_registry_status_label.setText("Downloading...")

        cache_root_txt = self.movie_cache_root_edit.text().strip() or None
        worker = _DatasetDownloadWorker(dataset_id=ds, cache_root=cache_root_txt)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        self._movie_download_thread = thread
        self._movie_download_worker = worker

        def _cleanup() -> None:
            thread.quit()
            worker.deleteLater()
            thread.deleteLater()
            self._movie_download_thread = None
            self._movie_download_worker = None

        def _on_ok(_path: str) -> None:
            self.movie_download_progress.setVisible(False)
            self._update_movie_registry_status()
            _cleanup()

        def _on_err(message: str) -> None:
            self.movie_download_progress.setVisible(False)
            self._update_movie_registry_status()
            try:
                box = QtWidgets.QMessageBox(self)
                box.setIcon(QtWidgets.QMessageBox.Critical)
                box.setWindowTitle("Dataset download failed")
                box.setText(str(message))
                box.setInformativeText(
                    "Try a local movie path, or choose a different registry dataset."
                )
                box.open()
            except Exception:
                pass
            _cleanup()

        thread.started.connect(worker.run)
        worker.finished.connect(_on_ok)
        worker.failed.connect(_on_err)
        thread.start()
