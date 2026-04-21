# Hydra Configuration Files

This directory contains YAML examples for the canonical `SimulationSpec` shape used by `create_simulation_spec(...)`.
The compatibility layer for nested component-style config mappings was removed, so configs should use the active top-level fields directly.

## Structure

```text
conf/
├── config.yaml                        # Base SimulationSpec-shaped defaults
└── experiment/
    ├── sparse_simple.yaml             # Baseline: sparse reward + discrete actions
    ├── dense_oriented.yaml            # Oriented actions + step-penalty reward
    ├── step_penalty_oriented.yaml     # Longer horizon step-penalty example
    └── antennae_array.yaml            # Antennae observation example
```

## Canonical Fields

These YAML files map directly onto `plume_nav_sim.config.SimulationSpec`:

- `grid_size`
- `source_location`
- `start_location`
- `max_steps`
- `goal_radius`
- `plume_sigma`
- `action_type`
- `step_size`
- `observation_type`
- `reward_type`
- `render`
- `plume`
- `movie_path`, `movie_dataset_id`, `movie_auto_download`, `movie_*`

Use `plume: movie` together with `movie_*` fields for movie-backed datasets. Do not nest settings under `action`, `observation`, `reward`, or `plume` sub-mappings.

## Usage

### Direct Loading

```python
from omegaconf import OmegaConf
from plume_nav_sim.config import build_env, create_simulation_spec

loaded = OmegaConf.to_container(
    OmegaConf.load("conf/experiment/sparse_simple.yaml"),
    resolve=True,
)
spec = create_simulation_spec(loaded)
env = build_env(spec)
```

### Hydra Decorator

```python
import hydra
from omegaconf import DictConfig, OmegaConf
from plume_nav_sim.config import build_env, create_simulation_spec

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    loaded = OmegaConf.to_container(cfg, resolve=True)
    spec = create_simulation_spec(loaded)
    env = build_env(spec)
    obs, info = env.reset()
    env.close()

if __name__ == "__main__":
    main()
```

Run with:

```bash
python train.py experiment=sparse_simple
python train.py experiment=step_penalty_oriented grid_size=[256,256]
```

### Programmatic Override

```python
from hydra import compose, initialize
from omegaconf import OmegaConf
from plume_nav_sim.config import build_env, create_simulation_spec

with initialize(version_base=None, config_path="conf"):
    cfg = compose(
        config_name="config",
        overrides=[
            "experiment=sparse_simple",
            "max_steps=2000",
            "step_size=2",
        ],
    )
    spec = create_simulation_spec(OmegaConf.to_container(cfg, resolve=True))
    env = build_env(spec)
```

## Example Custom Config

```yaml
# conf/experiment/my_experiment.yaml
# @package _global_

grid_size: [256, 256]
source_location: [200, 200]
max_steps: 5000
goal_radius: 15.0
plume_sigma: 30.0
action_type: oriented
step_size: 3
observation_type: antennae
reward_type: step_penalty
render: false
```

## Validation

Use the lightweight canonical environment config when you only need env-init validation:

```python
from plume_nav_sim import create_environment_config

config = create_environment_config(
    {
        "grid_size": [128, 128],
        "source_location": [64, 64],
        "max_steps": 1000,
        "goal_radius": 5.0,
        "plume_params": {"sigma": 20.0},
    }
)
```

Use `create_simulation_spec(...)` when validating the full selector-based runtime config.

## Best Practices

1. Use experiment configs as small overrides over `config.yaml`.
2. Validate configs before starting long runs.
3. Keep movie plume settings in `movie_*` fields so they round-trip cleanly through `SimulationSpec`.
4. Prefer `prepare(...)` or `build_env(...)` over ad hoc constructor branching in application code.
5. Commit config changes alongside code and test updates.

## See Also

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- `plume_nav_sim.config.SimulationSpec`
- `plume_nav_sim.EnvironmentConfig`
- External plug-and-play, spec-first usage:
  - `plug-and-play-demo/README.md` (section: "Config-based composition")
  - `plug-and-play-demo/configs/simulation_spec.json`
