from __future__ import annotations

import gzip
import json
import os
import platform
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .schemas import EpisodeRecord, RunMeta, StepRecord


class _RotatingJsonlGzWriter:
    def __init__(self, base_path: Path, rotate_size_bytes: Optional[int]) -> None:
        self._base_path = base_path
        self._rotate_size_bytes = rotate_size_bytes
        self._part = 0
        self._current_path = base_path
        self._fh = self._open_path(base_path)

    def _open_path(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        return gzip.open(path, "at", encoding="utf-8")

    def _next_path(self) -> Path:
        self._part += 1
        return self._base_path.with_name(
            f"{self._base_path.stem}.part{self._part}{self._base_path.suffix}"
        )

    def _should_rotate(self) -> bool:
        if not self._rotate_size_bytes:
            return False
        try:
            return self._current_path.exists() and self._current_path.stat().st_size >= int(
                self._rotate_size_bytes
            )
        except Exception:
            return False

    def write_obj(self, obj: dict) -> None:
        if self._should_rotate():
            self._fh.close()
            self._current_path = self._next_path()
            self._fh = self._open_path(self._current_path)
        self._fh.write(json.dumps(obj, sort_keys=True))
        self._fh.write("\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class RunRecorder:
    def __init__(
        self,
        root_dir: str | Path,
        *,
        experiment: Optional[str] = None,
        run_id: Optional[str] = None,
        rotate_size_bytes: Optional[int] = None,
    ) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        self.run_id = run_id or f"run-{ts}"
        self.experiment = experiment or "default"
        self.root = Path(root_dir) / self.experiment / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)

        self._steps = _RotatingJsonlGzWriter(
            self.root / "steps.jsonl.gz", rotate_size_bytes
        )
        self._episodes = _RotatingJsonlGzWriter(
            self.root / "episodes.jsonl.gz", rotate_size_bytes
        )
        self._meta_path = self.root / "run.json"

    def write_run_meta(self, meta: RunMeta) -> None:
        payload = meta.to_dict()
        if payload.get("system") is None:
            payload["system"] = {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": sys.version.split()[0],
                "pid": os.getpid(),
                "user": os.environ.get("USER") or os.environ.get("USERNAME"),
            }
        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def append_step(self, rec: StepRecord) -> None:
        self._steps.write_obj(asdict(rec))

    def append_episode(self, rec: EpisodeRecord) -> None:
        self._episodes.write_obj(asdict(rec))

    def finalize(self, export_parquet: bool = False) -> None:
        if export_parquet:
            raise ValueError("Parquet export is no longer supported in data_capture")
        self._steps.close()
        self._episodes.close()
