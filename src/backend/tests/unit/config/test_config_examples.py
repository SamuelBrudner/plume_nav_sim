from __future__ import annotations

from pathlib import Path

import yaml

from plume_nav_sim.config.component_configs import ComponentEnvironmentConfig


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


def test_base_config_yaml_matches_environment_config_model() -> None:
    loaded = _load_yaml(BASE_CONFIG)
    loaded.pop("defaults", None)

    validated = ComponentEnvironmentConfig.model_validate(loaded)

    assert validated.reward.goal_reward == 1.0
    assert validated.reward.step_penalty == 0.01


def test_step_penalty_experiment_yaml_matches_environment_config_model() -> None:
    loaded = _load_yaml(STEP_PENALTY_EXPERIMENT)

    validated = ComponentEnvironmentConfig.model_validate(loaded)

    assert validated.reward.goal_reward == 10.0
    assert validated.reward.step_penalty == 0.01


def test_data_capture_configs_do_not_advertise_render_flag() -> None:
    base = _load_yaml(DATA_CAPTURE_CONFIG)
    experiment = _load_yaml(DATA_CAPTURE_EXPERIMENT)

    assert "render" not in base["env"]
    assert "render" not in experiment["env"]
