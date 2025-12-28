from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional, Tuple


def run_small_capture(
    out_root: Path,
    *,
    experiment: str = "ci_guard",
    episodes: int = 2,
    grid: Tuple[int, int] = (8, 8),
    max_steps: Optional[int] = 20,
    seed: int = 123,
    export_parquet: bool = True,
) -> Path:
    """Run a tiny deterministic capture via the CLI and return the run dir.

    The function imports and calls the CLI entrypoint directly to avoid PATH
    issues on CI runners. It then discovers the created run directory and
    returns it for downstream validation.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Build CLI argv
    grid_str = f"{int(grid[0])}x{int(grid[1])}"
    argv = [
        "--output",
        str(out_root),
        "--experiment",
        experiment,
        "--episodes",
        str(int(episodes)),
        "--grid",
        grid_str,
        "--seed",
        str(int(seed)),
    ]
    if max_steps is not None:
        argv += ["--max-steps", str(int(max_steps))]
    if export_parquet:
        argv += ["--parquet"]

    # Prefer direct import of CLI for reliability
    mod = importlib.import_module("plume_nav_sim.cli.capture")
    rc = int(mod.main(argv))
    if rc != 0:
        raise RuntimeError(f"plume-nav-capture exited with code {rc}")

    # Identify the created run directory under out_root/experiment
    exp_dir = out_root / experiment
    candidates = [
        p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("run-")
    ]
    if not candidates:
        raise RuntimeError(f"No run directory found in {exp_dir}")
    # Pick the most recently modified directory
    run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    return run_dir
