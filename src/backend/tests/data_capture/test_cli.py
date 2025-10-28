from pathlib import Path

from plume_nav_sim.cli.capture import main as capture_main


def test_capture_cli_runs_and_writes(tmp_path: Path):
    out = tmp_path / "out"
    code = capture_main(
        [
            "--output",
            str(out),
            "--experiment",
            "cli-test",
            "--episodes",
            "2",
            "--seed",
            "100",
            "--grid",
            "8x8",
        ]
    )
    assert code == 0
    run_dirs = list((out / "cli-test").glob("run-*/"))
    assert run_dirs, "Expected run directory created"
    steps = run_dirs[0] / "steps.jsonl.gz"
    episodes = run_dirs[0] / "episodes.jsonl.gz"
    assert steps.exists(), "steps.jsonl.gz missing"
    assert episodes.exists(), "episodes.jsonl.gz missing"
