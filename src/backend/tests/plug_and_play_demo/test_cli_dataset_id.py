import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_demo_main() -> object:
    root = Path(__file__).resolve().parents[4]
    script_path = root / "plug-and-play-demo" / "main.py"
    spec = importlib.util.spec_from_file_location(
        "plug_and_play_demo_main", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load plug-and-play demo module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_config_prefers_cli_dataset_id_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo_main = _load_demo_main()
    config = {
        "simulation": {
            "plume": "movie",
            "movie_dataset_id": "config_dataset",
            "render": False,
        }
    }
    cfg_path = tmp_path / "sim.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")

    captured = {}

    def fake_prepare(sim):
        captured["sim"] = sim
        return object(), object()

    class DummyResult:
        def __init__(self, seed: int) -> None:
            self.seed = seed
            self.steps = 0
            self.total_reward = 0.0
            self.terminated = False
            self.truncated = False

    def fake_run_episode(env, policy, seed, render, on_step):
        return DummyResult(seed)

    monkeypatch.setattr(demo_main, "prepare", fake_prepare)
    monkeypatch.setattr(demo_main.runner, "run_episode", fake_run_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo",
            "--config",
            str(cfg_path),
            "--movie-dataset-id",
            "cli_dataset",
            "--no-render",
        ],
    )

    demo_main.main()

    sim = captured["sim"]
    assert sim.movie_dataset_id == "cli_dataset"
    assert sim.plume == "movie"
