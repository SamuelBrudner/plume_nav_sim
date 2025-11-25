from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from plume_nav_sim.cli.capture import (
    _env_from_cfg,
    _load_hydra_config,
    _manual_compose_for_data_capture,
)
from plume_nav_sim.cli.capture import main as capture_main


def _latest_run_dir(root: Path, experiment: str) -> Path:
    exp_dir = root / experiment
    candidates = [
        p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("run-")
    ]
    assert candidates, f"No run directory found in {exp_dir}"
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _count_jsonl_gz(path: Path) -> int:
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def test_hydra_cli_resolves_and_propagates_hash(tmp_path: Path):
    out = tmp_path / "out"
    exp = "hydra-cli"
    # Run CLI with Hydra config and overrides
    argv = [
        "--output",
        str(out),
        "--experiment",
        exp,
        "--seed",
        "200",
        "--config-name",
        "data_capture/config",
        # overrides captured via parse_known_args
        "episodes=2",
        "env.grid_size=[8,8]",
        "env.max_steps=10",
    ]
    code = capture_main(argv)
    assert code == 0

    # Discover run dir and artifacts
    run_dir = _latest_run_dir(out, exp)
    run_meta_path = run_dir / "run.json"
    manifest_path = run_dir / "manifest.json"
    episodes_path = run_dir / "episodes.jsonl.gz"

    assert run_meta_path.exists(), "run.json missing"
    assert manifest_path.exists(), "manifest.json missing"
    assert episodes_path.exists(), "episodes.jsonl.gz missing"

    # Run meta assertions
    with open(run_meta_path, "r", encoding="utf-8") as fh:
        run_meta = json.load(fh)

    # 1) Resolved config matches expected env settings (grid size)
    assert run_meta["env_config"]["grid_size"] == [8, 8]

    # Episodes override effective: episodes.jsonl.gz should have 2 records
    assert _count_jsonl_gz(episodes_path) == 2

    # 2) config_hash equals SHA256 of resolved Hydra config JSON (from helper)
    cfg_resolved, expected_hash = _load_hydra_config(
        config_name="data_capture/config",
        config_path=None,
        overrides=["episodes=2", "env.grid_size=[8,8]", "env.max_steps=10"],
    )
    assert isinstance(cfg_resolved, dict)
    assert run_meta.get("config_hash") == expected_hash

    # 3) Manifest includes config_hash and it equals SHA256 of env_config JSON
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest_hash = manifest.get("config_hash")
    assert isinstance(manifest_hash, str) and len(manifest_hash) == 64

    env_cfg_bytes = json.dumps(
        run_meta.get("env_config"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    env_cfg_hash = hashlib.sha256(env_cfg_bytes).hexdigest()
    assert manifest_hash == env_cfg_hash


def test_hydra_cli_malformed_grid_fails(tmp_path: Path):
    out = tmp_path / "out"
    exp = "hydra-bad-grid"
    # Passing a malformed grid (single value) should error
    bad_argv = [
        "--output",
        str(out),
        "--experiment",
        exp,
        "--config-name",
        "data_capture/config",
        "env.grid_size=[8]",
    ]
    with pytest.raises(SystemExit):
        _ = capture_main(bad_argv)


def test_env_from_cfg_wires_dataset_registry(monkeypatch, tmp_path: Path):
    import plume_nav_sim as pns

    captured: dict[str, object] = {}

    class DummyEnv:
        def __init__(self) -> None:
            self.grid_size = (12, 10)

    monkeypatch.setattr(
        pns, "make_env", lambda **kwargs: captured.update(kwargs) or DummyEnv()
    )

    cfg = {
        "env": {
            "plume": "movie",
        },
        "movie": {
            "dataset_id": "colorado_jet_v1",
            "auto_download": True,
            "cache_root": str(tmp_path / "cache"),
            "path": str(tmp_path / "override.zarr"),
            "fps": 9.5,
        },
    }

    env, w, h = _env_from_cfg(cfg, action_type_override=None)

    assert isinstance(env, DummyEnv)
    assert (w, h) == (12, 10)
    assert captured.get("movie_dataset_id") == "colorado_jet_v1"
    assert captured.get("movie_auto_download") is True
    assert captured.get("movie_cache_root") == str(tmp_path / "cache")
    assert captured.get("movie_path") == str(tmp_path / "override.zarr")
    assert captured.get("movie_fps") == 9.5


def test_manual_compose_matches_hydra(tmp_path: Path):
    """Manual data_capture composition should match Hydra's resolved config.

    This protects the behavior of _manual_compose_for_data_capture when
    resolving mixed-group configs such as data_capture/config.
    """
    hydra = pytest.importorskip("hydra")
    omegaconf = pytest.importorskip("omegaconf")

    import plume_nav_sim as pns

    config_name = "data_capture/config"
    overrides = ["episodes=3", "env.grid_size=[8,8]", "env.max_steps=5"]

    conf_dir = pns.get_conf_dir()
    assert conf_dir is not None
    conf_root = Path(conf_dir)

    manual_cfg = _manual_compose_for_data_capture(
        config_name=config_name,
        conf_root=conf_root,
        overrides=list(overrides),
    )
    hydra_cfg, _ = _load_hydra_config(
        config_name=config_name,
        config_path=None,
        overrides=list(overrides),
    )

    # On failure, show the full resolved configs for manual vs Hydra
    assert manual_cfg == hydra_cfg, (
        "\nmanual_cfg="
        + json.dumps(manual_cfg, sort_keys=True, indent=2)
        + "\nhydra_cfg="
        + json.dumps(hydra_cfg, sort_keys=True, indent=2)
    )
