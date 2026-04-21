from __future__ import annotations

from pathlib import Path

import yaml

from plume_nav_sim.config import create_simulation_spec


REPO_ROOT = Path(__file__).resolve().parents[5]
BASE_CONFIG = REPO_ROOT / "src" / "backend" / "conf" / "config.yaml"
STEP_PENALTY_EXPERIMENT = (
    REPO_ROOT / "src" / "backend" / "conf" / "experiment" / "step_penalty_oriented.yaml"
)
DATA_CAPTURE_CONFIG = REPO_ROOT / "src" / "backend" / "conf" / "data_capture" / "config.yaml"
DATA_CAPTURE_EXPERIMENT = (
    REPO_ROOT
    / "src"
    / "backend"
    / "conf"
    / "data_capture"
    / "experiment"
    / "default.yaml"
)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def test_base_config_yaml_translates_to_simulation_spec() -> None:
    loaded = _load_yaml(BASE_CONFIG)
    spec = create_simulation_spec(loaded)

    assert spec.grid_size == (128, 128)
    assert spec.source_location == (64, 64)
    assert spec.action_type == "discrete"
    assert spec.step_size == 1
    assert spec.reward_type == "sparse"
    assert spec.goal_radius == 5.0
    assert spec.plume_sigma == 20.0
    assert spec.render is False


def test_step_penalty_experiment_yaml_translates_to_simulation_spec() -> None:
    loaded = _load_yaml(STEP_PENALTY_EXPERIMENT)
    spec = create_simulation_spec(loaded)

    assert spec.grid_size == (128, 128)
    assert spec.source_location == (100, 100)
    assert spec.action_type == "oriented"
    assert spec.step_size == 2
    assert spec.reward_type == "step_penalty"
    assert spec.goal_radius == 10.0
    assert spec.plume_sigma == 25.0


def test_data_capture_configs_do_not_advertise_render_flag() -> None:
    base = _load_yaml(DATA_CAPTURE_CONFIG)
    experiment = _load_yaml(DATA_CAPTURE_EXPERIMENT)

    assert "render" not in base["env"]
    assert "render" not in experiment["env"]
