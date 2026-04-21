# Extending `plume_nav_sim`

PlumeNav’s dependency-injection stack lets you plug in custom components without touching the core environment. Use this guide when you are ready to move beyond `make_env()` and ship research-grade extensions.

For an overview of the public API surface and where code lives in the repository, see:

- `src/backend/README.md` – section "Public API and Repository Layout".
- `src/backend/CONTRIBUTING.md` – section "Repository Layout and Public API".

## When to Extend

- **New sensing modalities**: Swap in a custom `ObservationModel` for camera feeds, multi-sensor arrays, or synthetic sensors.
- **Alternative control policies**: Provide a bespoke `ActionProcessor` to experiment with hybrid or continuous controls.
- **Reward shaping**: Implement domain-specific `RewardFunction` logic while keeping Gymnasium compatibility.
- **Plume physics**: Prototype new plume generators by wiring alternative `ConcentrationField` implementations.

## Extension Roadmap

- **Protocol Interfaces** → `docs/extending/protocol_interfaces.md`
  - Contract summaries for `ActionProcessor`, `ObservationModel`, and `RewardFunction`.
  - Type expectations, required methods, and validation semantics.

- **Component Injection Guide** → `docs/extending/component_injection.md`
  - How DI wiring works inside `PlumeEnv`.
  - Manually assembling environments and registering them with Gymnasium.
  - Real example: `src/backend/examples/custom_components.py`.

- **Custom Component Tutorials**
  - Rewards → `docs/extending/custom_rewards.md`
  - Observations → `docs/extending/custom_observations.md`
  - Actions → `docs/extending/custom_actions.md`

- **Configuration & Factories**
  - API surface: `plume_nav_sim/envs/factory.py`, `plume_nav_sim/config`.
  - Sample configs: `conf/experiment/*.yaml`.
  - Migration history: `docs/MIGRATION_COMPONENT_ENV.md`.

## Quick Start

```python
import plume_nav_sim as pns
from plume_nav_sim.envs import PlumeEnv

env = pns.make_env()
base = env.unwrapped  # PlumeEnv
env.close()

# Assemble your own environment
custom_env = PlumeEnv(
    grid_size=base._grid_size,
    source_location=base.goal_location,
    goal_radius=base.goal_radius,
    plume=my_plume_model,
    action_model=my_action_processor,
    sensor_model=my_observation_model,
    reward_fn=my_reward_function,
)
```

See `src/backend/examples/custom_components.py` for a full walkthrough, including reproducible rollouts and metadata collection.

## Best Practices

- **Duck typing**: Satisfy protocol shapes; inheritance is optional.
- **Validation**: Reuse helpers in `plume_nav_sim._compat` for consistent error reporting.
- **Testing**: Start with `tests/contracts/` suites to harden new components before integration.
- **Documentation**: Record behavioral differences in `docs/extending/` so future experiments build on verified groundwork.
