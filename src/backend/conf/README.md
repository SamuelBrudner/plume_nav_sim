# Hydra Configuration Files

This directory contains YAML configuration files for creating environments using [Hydra](https://hydra.cc/).

## Structure

```text
conf/
├── config.yaml              # Base config with defaults
└── experiment/              # Experiment-specific configs
    ├── sparse_simple.yaml   # Baseline: sparse reward + discrete actions
    ├── dense_oriented.yaml  # Shaped reward + orientation
    └── antennae_array.yaml  # Multi-sensor observation
```

## Usage

### Method 1: Direct Loading (Recommended for Scripts)

```python
from omegaconf import OmegaConf
from plume_nav_sim.config import EnvironmentConfig, create_environment_from_config

# Load YAML
cfg_dict = OmegaConf.to_container(OmegaConf.load("conf/experiment/sparse_simple.yaml"))

# Parse into Pydantic model (validates!)
config = EnvironmentConfig(**cfg_dict)

# Create environment
env = create_environment_from_config(config)
```

### Method 2: Hydra Decorator (Recommended for Applications)

```python
import hydra
from omegaconf import DictConfig
from plume_nav_sim.config import EnvironmentConfig, create_environment_from_config

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Convert to Pydantic (validates)
    config = EnvironmentConfig(**cfg)
    
    # Create environment
    env = create_environment_from_config(config)
    
    # Your training loop...
    obs, info = env.reset()
    # ...

if __name__ == "__main__":
    main()
```

Run with:

```bash
python train.py experiment=sparse_simple
python train.py experiment=dense_oriented grid_size=[256,256]
```

### Method 3: Programmatic Override

```python
from hydra import compose, initialize
from plume_nav_sim.config import EnvironmentConfig, create_environment_from_config

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config", overrides=[
        "experiment=sparse_simple",
        "max_steps=2000",
        "action.step_size=2"
    ])
    
    config = EnvironmentConfig(**cfg)
    env = create_environment_from_config(config)
```

## Configuration Options

### Environment Settings

- `grid_size`: `[width, height]` - Environment dimensions
- `goal_location`: `[x, y]` - Target position
- `start_location`: `[x, y]` or `null` - Initial position (null = center)
- `max_steps`: int - Episode step limit
- `render_mode`: `"rgb_array"`, `"human"`, or `null`

### Action Processor (`action`)

- `type`: `"discrete"` (4-dir) or `"oriented"` (3-action)
- `step_size`: int - Movement distance per step

### Observation Model (`observation`)

- `type`: `"concentration"` (single) or `"antennae"` (array)
- `n_sensors`: int - Number of sensors (antennae only)
- `sensor_distance`: float - Distance from agent (antennae only)
- `sensor_angles`: list[float] or null - Custom angles (antennae only)

### Reward Function (`reward`)

- `type`: `"sparse"` (binary) or `"dense"` (shaped)
- `goal_radius`: float - Success threshold distance
- `distance_weight`: float - Weight for distance term (dense only)
- `concentration_weight`: float - Weight for concentration term (dense only)

### Plume Field (`plume`)

- `sigma`: float - Gaussian dispersion parameter
- `normalize`: bool - Normalize to [0, 1]
- `enable_caching`: bool - Cache concentration lookups

## Creating Custom Configs

1. Copy an existing experiment config
2. Modify parameters
3. Save to `conf/experiment/my_experiment.yaml`
4. Use with `experiment=my_experiment`

Example custom config:
```yaml
# conf/experiment/my_experiment.yaml
# @package _global_

grid_size: [256, 256]
goal_location: [200, 200]
max_steps: 5000

action:
  type: discrete
  step_size: 3

observation:
  type: antennae
  n_sensors: 6
  sensor_distance: 3.0

reward:
  type: dense
  goal_radius: 15.0

plume:
  sigma: 30.0
```

## Validation

Pydantic validates all configs automatically:
```python
config = EnvironmentConfig(**cfg)  # Raises ValidationError if invalid
```

Common validation errors:
- `goal_location` outside `grid_size` bounds
- Negative or zero `step_size`, `sigma`, `goal_radius`
- Invalid `type` values
- `sensor_angles` length doesn't match `n_sensors`

## Best Practices

1. **Use experiment configs**: Override base `config.yaml` with experiment-specific settings
2. **Validate early**: Load config at script start to catch errors immediately
3. **Version control**: Commit config files alongside code
4. **Experiment tracking**: Log the full config with your experiment results
5. **Sweeps**: Use Hydra multirun for hyperparameter sweeps

Example sweep:
```bash
python train.py -m experiment=sparse_simple action.step_size=1,2,3 plume.sigma=10,20,30
```

## See Also

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- `plume_nav_sim/config/component_configs.py` - Pydantic model definitions
- `docs/MIGRATION_COMPONENT_ENV.md` - Component-based environment guide
