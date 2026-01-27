from __future__ import annotations

import atexit
import gzip
import io
import os
import threading
from pathlib import Path
from typing import Any, Optional

try:
    import orjson as _orjson
except Exception:  # pragma: no cover
    _orjson = None  # type: ignore
import json as _json


def _dumps(obj: Any) -> bytes:
    if _orjson is not None:
        return _orjson.dumps(obj)
    # Fallback: ensure ASCII-safe, no spaces, then encode
    return _json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class JSONLGzWriter:
    def __init__(
        self,
        path: os.PathLike | str,
        *,
        rotate_size_bytes: Optional[int] = None,
        compresslevel: int = 6,
        buffer_size: int = 64 * 1024,
    ) -> None:
        self.base_path = Path(path)
        self.rotate_size_bytes = int(rotate_size_bytes) if rotate_size_bytes else None
        self.compresslevel = int(compresslevel)
        self.buffer_size = int(buffer_size)

        self._lock = threading.Lock()
        self._fh: Optional[gzip.GzipFile] = None
        self._buf: Optional[io.BufferedWriter] = None
        self._current_path: Optional[Path] = None
        self._part = 0

        self._open_new_file()
        atexit.register(self.close)

    def _current_size(self) -> int:
        try:
            if self._current_path and self._current_path.exists():
                return self._current_path.stat().st_size
        except Exception:
            return 0
        return 0

    def _open_new_file(self) -> None:
        if not self.base_path.parent.exists():
            self.base_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = f".part{self._part:04d}.jsonl.gz" if self._part > 0 else ".jsonl.gz"
        target = self.base_path.with_suffix(
            self.base_path.suffix + suffix if self.base_path.suffix else suffix
        )
        # If base_path has extension already, append .jsonl.gz to it; else create .jsonl.gz
        if self.base_path.suffix:
            # Normalize: name.ext + .jsonl.gz
            target = Path(str(self.base_path) + suffix)

        raw = open(target, "wb", buffering=0)
        gz = gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=self.compresslevel)
        buf = io.BufferedWriter(gz, buffer_size=self.buffer_size)

        self._fh = gz
        self._buf = buf
        self._current_path = target

    def _rotate_if_needed(self) -> None:
        if self.rotate_size_bytes is None:
            return
        size = self._current_size()
        if size >= self.rotate_size_bytes:
            # Close current file and open next part
            self._buf.flush()  # type: ignore[union-attr]
            self._fh.close()  # type: ignore[union-attr]
            self._part += 1
            self._open_new_file()

    def write_obj(self, obj: Any) -> None:
        line = _dumps(obj) + b"\n"
        with self._lock:
            self._buf.write(line)  # type: ignore[union-attr]
            # Heuristic: flush buffer if it is near capacity
            if self._buf and self._buf.raw and getattr(self._buf.raw, "fileobj", None):
                pass  # leave buffered; rely on rotation/close
            self._rotate_if_needed()

    def flush(self) -> None:
        with self._lock:
            try:
                if self._buf:
                    self._buf.flush()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            try:
                if self._buf:
                    self._buf.flush()
                if self._fh:
                    self._fh.close()
            except Exception:
                pass
            finally:
                self._buf = None
                self._fh = None
