from __future__ import annotations

import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import EpisodeRecord, RunMeta, StepRecord
from .validate import validate_run_artifacts
from .writer import JSONLGzWriter

_HAVE_PYARROW = None  # lazy detection


class RunRecorder:
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
        # Always generate a provenance manifest for this run
        try:
            self._write_manifest()
        except Exception:
            # Never fail user workflows due to manifest issues
            pass

    def _export_parquet(self) -> None:
        # Import lazily to keep import time minimal during normal operation
        import gzip
        import json

        # pandas is not required to be imported as a module here; the
        # flattened DataFrame objects returned by validators expose
        # .to_parquet(), which will use an available engine.
        # Reuse validation flatteners to maintain parity with schema
        from .validate import _flatten_episodes, _flatten_steps

        for stem in ("steps", "episodes"):
            jsonl_paths = sorted(self.root.glob(f"{stem}.part*.jsonl.gz"))
            base = self.root / f"{stem}.jsonl.gz"
            if base.exists():
                jsonl_paths.insert(0, base)
            if not jsonl_paths:
                continue

            rows = []
            for p in jsonl_paths:
                with gzip.open(p, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            rows.append(json.loads(line))
            if not rows:
                continue

            if stem == "steps":
                df = _flatten_steps(rows)
            else:
                df = _flatten_episodes(rows)

            if df is None or len(df) == 0:
                # Nothing to write
                continue

            # Write Parquet via pandas, using pyarrow engine when available
            out_path = self.root / f"{stem}.parquet"
            df.to_parquet(out_path, index=False)

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

    def _write_manifest(self) -> None:
        import json

        manifest = self._build_manifest_payload()

        with open(self.root / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)

    def _read_run_meta(self) -> Dict[str, Any]:
        import json

        try:
            with open(self._meta_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _git_sha(self) -> Optional[str]:
        import subprocess

        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(self.root.parent.parent),
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            return None

    def _package_version(self, run_meta: Dict[str, Any]) -> Optional[str]:
        pkg_version = run_meta.get("package_version")
        if pkg_version:
            return str(pkg_version)
        try:
            import importlib.metadata as _im

            return _im.version("plume_nav_sim")
        except Exception:
            return None

    def _config_hash(self, env_cfg: Any) -> Optional[str]:
        import hashlib
        import json

        if env_cfg is None:
            return None
        try:
            cfg_bytes = json.dumps(
                env_cfg, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            return hashlib.sha256(cfg_bytes).hexdigest()
        except Exception:
            return None

    def _file_inventory(self) -> list[Dict[str, Any]]:
        def _file_info(name: str) -> Dict[str, Any]:
            p = self.root / name
            try:
                st = p.stat()
                return {"path": name, "bytes": int(st.st_size)}
            except Exception:
                return {"path": name, "bytes": None}

        files: list[Dict[str, Any]] = []
        for fname in (
            "run.json",
            "steps.jsonl.gz",
            "episodes.jsonl.gz",
            "steps.parquet",
            "episodes.parquet",
        ):
            if (self.root / fname).exists():
                files.append(_file_info(fname))
        return files

    def _seed_summary(self) -> Dict[str, Any]:
        try:
            jsonl_paths = self._step_jsonl_paths()
            seeds = self._extract_seeds_from_files(jsonl_paths)
            return self._summarize_seeds(seeds)
        except Exception:
            return {}

    def _step_jsonl_paths(self) -> list[Path]:
        jsonl_paths = sorted(self.root.glob("steps.part*.jsonl.gz"))
        base = self.root / "steps.jsonl.gz"
        if base.exists():
            jsonl_paths.insert(0, base)
        return jsonl_paths

    @staticmethod
    def _summarize_seeds(seeds: set[int]) -> Dict[str, Any]:
        if not seeds:
            return {}
        return {
            "unique_count": len(seeds),
            "min": min(seeds),
            "max": max(seeds),
        }

    def _extract_seeds_from_files(self, paths: list[Path]) -> set[int]:
        import gzip as _gzip
        import json as _json

        seeds: set[int] = set()
        for p in paths:
            try:
                with _gzip.open(p, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        try:
                            obj = _json.loads(line)
                            seed = obj.get("seed")
                            if seed is not None:
                                seeds.add(int(seed))
                        except Exception:
                            continue
            except Exception:
                continue
        return seeds

    def _build_validation_report(self) -> Dict[str, Any]:
        try:
            return validate_run_artifacts(self.root)
        except Exception as e:
            return {"error": str(e)}

    def _schema_version(self, run_meta: Dict[str, Any]) -> Optional[str]:
        try:
            return run_meta.get("schema_version")
        except Exception:
            return None

    def _build_manifest_payload(self) -> Dict[str, Any]:
        run_meta = self._read_run_meta()
        git_sha = self._git_sha()
        pkg_version = self._package_version(run_meta)
        env_cfg = run_meta.get("env_config")
        cfg_hash = self._config_hash(env_cfg)
        report = self._build_validation_report()
        schema_version = self._schema_version(run_meta)

        manifest: Dict[str, Any] = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha,
            "package_version": pkg_version,
            "schema_version": schema_version,
            "config_hash": cfg_hash,
            "env_config": env_cfg,
            "validation": report,
            "files": self._file_inventory(),
            "seed_summary": self._seed_summary(),
        }

        return manifest
