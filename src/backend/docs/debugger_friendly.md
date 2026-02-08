# Make Your App Debugger-Friendly

## Why implement a `DebuggerProvider`?
- Show stable, human-readable action labels instead of raw action indices.
- Show policy distributions (`probs`, `q_values`, or `logits`) in the inspector.
- Add observation metadata (`kind`, `label`) so the UI chooses better previews.
- Keep debugger behavior deterministic and app-specific without debugger code changes.

## How to make your app debugger-friendly

### 1) Implement a provider (subclass or duck-type)
Implement any subset of ODC methods (`get_action_info`, `describe_observation`, `policy_distribution`, `get_pipeline`). Return `None` for unsupported parts.

```python
from typing import Any
from plume_nav_debugger.odc.models import ActionInfo
from plume_nav_debugger.odc.provider import DebuggerProvider

class MyProvider(DebuggerProvider):
    """Minimal provider with custom action names."""

    def get_action_info(self, env: Any):
        return ActionInfo(
            names=[
                "FORWARD",
                "TURN_LEFT",
                "TURN_RIGHT",
            ]
        )
```

### 2) Register via `pyproject.toml` entry point (recommended)

```toml
[project.entry-points."plume_nav_sim.debugger_plugins"]
my_app = "my_app.debugger:provider_factory"
# provider_factory(env, policy) -> provider
```

### 3) Or expose provider directly on env/policy
If you do not install an entry point, expose a provider via reflection (checked on env, then policy):

```python
policy.__debugger_provider__ = MyProvider()
# also supported: env.__debugger_provider__, debugger_provider, get_debugger_provider()
```

## Testing your provider quickly

```bash
python - <<'PY'
from my_app.debugger import MyProvider
p = MyProvider()
print(p.get_action_info(env=object()))
print(p.describe_observation([0.1, 0.2], context={}))
print(p.policy_distribution(policy=object(), observation=[0.1, 0.2]))
print(p.get_pipeline(env=object()))
PY
```

If methods are optional in your provider, returning `None` is valid.

## References
- ODC spec: `src/plume_nav_debugger/odc/SPEC.md`
- Protocol/API: `src/plume_nav_debugger/odc/provider.py`
- Complete working example: `plug-and-play-demo/`
