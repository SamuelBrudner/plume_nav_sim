# Extending plume_nav_sim: Public API Overview

This index links the main extension guides and shows where to start for
protocol-based components, dependency injection, and config-driven setup.

- Protocol Interfaces (start here if you’re new to protocols)
  - docs/extending/protocol_interfaces.md
  - Explains Python Protocols, duck typing, and how our interfaces work at type and runtime.

- Component Injection (how to wire components into an env)
  - docs/extending/component_injection.md
  - Shows factory assembly, Gym registration with components, and manual wiring.
  - See also runnable example: examples/component_di_usage.py

- Create Custom Components
  - Rewards: docs/extending/custom_rewards.md
  - Observations: docs/extending/custom_observations.md
  - Actions: docs/extending/custom_actions.md

- Configuration and Factories
  - Code: plume_nav_sim/config/factories.py
  - Example configs: conf/README.md, conf/experiment/*.yaml
  - Migration and patterns: docs/MIGRATION_COMPONENT_ENV.md

Tips
- Implementations only need to conform to interface shape (duck typing); no inheritance required.
- Observation models consume an env_state dict assembled by the environment; see the env_state schema in the Component Injection guide.
- Spaces come from components: ActionProcessor.action_space and ObservationModel.observation_space.
- Test your components against the universal interface suites in tests/contracts/.
