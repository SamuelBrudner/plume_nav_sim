from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _have(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def test_docs_capture_cli_artifacts(tmp_path: Path):
    """Run plume-nav-capture per README and assert documented artifacts exist.

    Exercises CLI flags used in docs: --output, --experiment, --episodes, --grid, --parquet.
    """
    # Import CLI entry without requiring console script installation
    from plume_nav_sim.cli.capture import main as capture_main

    output_root = tmp_path / "results"
    experiment = "demo"

    # Minimal run per docs
    rc = capture_main(
        [
            "--output",
            str(output_root),
            "--experiment",
            experiment,
            "--episodes",
            "1",
            "--grid",
            "8x8",
            "--parquet",
        ]
    )
    assert rc == 0

    # Discover the run directory created by RunRecorder: results/<experiment>/<run_id>/
    exp_dir = output_root / experiment
    runs = [p for p in exp_dir.iterdir() if p.is_dir()]
    assert runs, f"No run directory found under {exp_dir}"
    run_dir = runs[0]

    # Core artifacts documented in README
    assert (run_dir / "run.json").exists(), "run.json missing"
    assert (run_dir / "steps.jsonl.gz").exists(), "steps.jsonl.gz missing"
    assert (run_dir / "episodes.jsonl.gz").exists(), "episodes.jsonl.gz missing"
    # Manifest is written when available
    assert (run_dir / "manifest.json").exists(), "manifest.json missing"

    # Parquet only asserted when dependencies are available
    if _have("pyarrow") and _have("pyarrow.parquet") and _have("pandas"):
        assert (run_dir / "steps.parquet").exists(), "steps.parquet missing"
        assert (run_dir / "episodes.parquet").exists(), "episodes.parquet missing"

    # Optional validation promise from README
    if _have("pandera") and _have("pandas"):
        from plume_nav_sim.data_capture.validate import validate_run_artifacts

        report = validate_run_artifacts(run_dir)
        assert report.get("steps", {}).get("ok") is True
        assert report.get("episodes", {}).get("ok") is True


def test_notebook_smoke_execution(tmp_path: Path):
    """Smoke-execute the plug-and-play demo notebook to guard against drift.

    Skips if notebook tooling is unavailable in the environment.
    """
    nbformat = pytest.importorskip("nbformat")
    nbconvert = pytest.importorskip("nbconvert")
    pytest.importorskip("ipykernel")

    repo_root = Path(__file__).resolve().parents[1]
    nb_path = repo_root / "plug-and-play-demo" / "plug_and_play_demo.ipynb"
    assert nb_path.exists(), f"Notebook not found: {nb_path}"

    # Load and prepend a sys.path bootstrap so imports work under pytest
    nb = nbformat.read(nb_path, as_version=4)
    bootstrap = nbformat.v4.new_code_cell(
        "from pathlib import Path\n"
        "import sys\n"
        "root = Path().resolve()\n"
        "src = root / 'src' / 'backend'\n"
        "demo = root / 'plug-and-play-demo'\n"
        "sys.path.insert(0, str(src))\n"
        "sys.path.insert(0, str(demo))\n"
        "print('Bootstrapped sys.path for notebook execution')\n"
    )
    nb.cells.insert(0, bootstrap)

    ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=120, kernel_name="python3")
    # Run in repo root to match relative paths in the notebook
    ep.preprocess(nb, {"metadata": {"path": str(repo_root)}})

    # Sanity: ensure our Capture Goals cell is present (docs from plume_nav_sim-161)
    assert any(
        getattr(c, "cell_type", "") == "markdown"
        and "Capture Goals and Workflow" in "".join(c.get("source", []))
        for c in nb.cells
    ), "Missing 'Capture Goals and Workflow' section in notebook"
