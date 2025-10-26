from __future__ import annotations

import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schemas import EpisodeRecord, RunMeta, StepRecord
from .writer import JSONLGzWriter

_HAVE_PYARROW = None  # lazy detection


class RunRecorder:
    """Manage a single run’s artifacts directory and record writing.

    Writes:
    - run.json (metadata)
    - steps.jsonl.gz (append)
    - episodes.jsonl.gz (append)
    Optional: export Parquet at finalize() if pyarrow available.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        experiment: Optional[str] = None,
        run_id: Optional[str] = None,
        rotate_size_bytes: Optional[int] = None,
    ) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_id = run_id or f"run-{ts}"
        self.experiment = experiment or "default"
        self.root = Path(root_dir) / self.experiment / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)

        self._steps = JSONLGzWriter(
            self.root / "steps", rotate_size_bytes=rotate_size_bytes
        )
        self._episodes = JSONLGzWriter(
            self.root / "episodes", rotate_size_bytes=rotate_size_bytes
        )
        self._meta_path = self.root / "run.json"

    def write_run_meta(self, meta: RunMeta) -> None:
        # Avoid large deps; write compact JSON manually
        import json

        # If system not set, enrich with system metadata
        sysinfo = meta.system
        if sysinfo.hostname is None:
            try:
                sysinfo = RunMeta.SystemInfo(
                    hostname=socket.gethostname(),
                    platform=platform.platform(),
                    python_version=sys.version.split()[0],
                    pid=os.getpid(),
                    user=os.environ.get("USER") or os.environ.get("USERNAME"),
                )
                meta = meta.model_copy(update={"system": sysinfo})
            except Exception:
                pass

        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta.model_dump(mode="json"), fh, indent=2, ensure_ascii=False)

    def append_step(self, rec: StepRecord) -> None:
        # Runtime validation ensured by Pydantic model instantiation
        self._steps.write_obj(rec.model_dump(mode="json"))

    def append_episode(self, rec: EpisodeRecord) -> None:
        self._episodes.write_obj(rec.model_dump(mode="json"))

    def finalize(self, export_parquet: bool = False) -> None:
        self._steps.close()
        self._episodes.close()
        if export_parquet and self._have_pyarrow():
            self._export_parquet()

    def _export_parquet(self) -> None:
        # Convert JSONL.gz to Parquet using pyarrow if present
        import pyarrow as _pa  # type: ignore
        import pyarrow.parquet as _pq  # type: ignore

        for stem in ("steps", "episodes"):
            jsonl_paths = sorted(self.root.glob(f"{stem}.part*.jsonl.gz"))
            base = self.root / f"{stem}.jsonl.gz"
            if base.exists():
                jsonl_paths.insert(0, base)
            if not jsonl_paths:
                continue

            rows = []
            import gzip
            import json

            for p in jsonl_paths:
                with gzip.open(p, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            rows.append(json.loads(line))
            if not rows:
                continue

            table = _pa.Table.from_pylist(rows)
            _pq.write_table(table, self.root / f"{stem}.parquet")

    def _have_pyarrow(self) -> bool:
        global _HAVE_PYARROW
        if _HAVE_PYARROW is not None:
            return bool(_HAVE_PYARROW)
        try:
            import importlib

            importlib.import_module("pyarrow")
            importlib.import_module("pyarrow.parquet")
            _HAVE_PYARROW = True
        except Exception:
            _HAVE_PYARROW = False
        return bool(_HAVE_PYARROW)
