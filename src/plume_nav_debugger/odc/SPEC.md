# Opinionated Debugger Contract (ODC)

Status: Draft 1.0.0

Goal: Enable the debugger to automatically and deterministically introspect user applications (custom actions, observations, policies, overlays) without requiring changes to debugger code, while remaining side‑effect free and reproducible.

Scope: This contract is optional. When present, the debugger prefers ODC. Heuristic fallbacks are not used; ODC is required for labels, distributions, and pipeline.

---

## Provider Interface (Python)

Implement a `DebuggerProvider` with the following optional methods. The debugger calls only the methods it needs; absence means “no opinion”. All methods must be side‑effect free.

```python
from typing import Any, Optional

class DebuggerProvider:
    # Actions --------------------------------------------------------------
    def get_action_info(self, env: Any) -> Optional[ActionInfo]:
        """Return names for action indices (0..n-1)."""

    # Observations ---------------------------------------------------------
    def describe_observation(
        self, observation: Any, *, context: Optional[dict] = None
    ) -> Optional[ObservationInfo]:
        """Describe the policy observation for presentation (kind, label)."""

    # Policy distribution --------------------------------------------------
    def policy_distribution(self, policy: Any, observation: Any) -> Optional[dict]:
        """Return one of: {"probs" | "q_values" | "logits"}: list[float]."""

    # Pipeline -------------------------------------------------------------
    def get_pipeline(self, env: Any) -> Optional[PipelineInfo]:
        """Return ordered names of wrappers/components: [Top, ..., Core]."""
```

Notes:
- Side‑effect free: Never call `select_action`, never mutate policy/env state.
- Determinism: For a given `(policy, observation)`, return the same distribution.
- Partial capability: Return `None` to decline; the debugger will hide that detail.

## Data Models

Python dataclasses (in `plume_nav_debugger.odc.models`) used at runtime:
- `ActionInfo { names: list[str] }`
- `ObservationInfo { kind: str, label?: str }` where `kind ∈ {"vector","image","scalar","unknown"}`
- `PipelineInfo { names: list[str] }`

Pydantic schemas (in `plume_nav_debugger.odc.schemas`) exist for JSON interchange and future remote attach.

## Discovery (How the debugger finds a provider)

Priority order:
1. Entry point plugins (recommended)
2. Reflection on `env` and `policy`

### 1) Entry point plugin

Group: `plume_nav_sim.debugger_plugins`

Each entry provides either:
- a factory `callable(env, policy) -> DebuggerProvider`, or
- a `DebuggerProvider` instance (no‑arg constructible class or object)

Example `pyproject.toml`:
```toml
[project.entry-points."plume_nav_sim.debugger_plugins"]
my_app = "my_app.debugger:provider_factory"
```

Example `my_app/debugger.py`:
```python
from plume_nav_debugger.odc.provider import DebuggerProvider
from plume_nav_debugger.odc.models import ActionInfo, PipelineInfo

class MyProvider(DebuggerProvider):
    def get_action_info(self, env):
        return ActionInfo(names=["RUN", "TUMBLE"])  # len == action_space.n

    def policy_distribution(self, policy, observation):
        # No side effects. Return one key with a 1D list of floats.
        return {"probs": [0.8, 0.2]}

    def get_pipeline(self, env):
        return PipelineInfo(names=[type(env).__name__, "MyWrapper", "CoreEnv"])

def provider_factory(env, policy):
    return MyProvider()
```

### 2) Reflection fallback

When no entry point is found, the debugger checks these in order on `env`, then `policy`:
- `get_debugger_provider()`
- `__debugger_provider__`
- `debugger_provider`

If any returns a `DebuggerProvider`, it is used.

## Semantics / Invariants

Actions:
- `get_action_info().names` length must match the policy/env action space cardinality.
- Names are stable across a run and index‑aligned (0..n‑1).

Distribution:
- Must be side‑effect free.
- Exactly one of `probs`, `q_values`, `logits` present.
- Shapes are 1D and length equals the number of actions.
- `probs`: normalized to sum to 1.0; the debugger will normalize again defensively.
- `q_values` / `logits`: debugger applies softmax to produce a probability preview.

Pipeline:
- `names` is ordered outermost → innermost (Top wrapper to Core env).
- Use descriptive names; include parameters where helpful, e.g., `"ConcentrationNBackWrapper(n=3)"`.

Observation:
- `kind` guides visualization (vector → sparkline, image → thumbnail, scalar → value).
- `label` appears in UI as a compact descriptor.

## Versioning / Handshake

For now, version is implicit (1.0.0). Providers may optionally expose `odc_version: str`. Future versions may add a handshake method.

## Strict Mode

- Strict provider-only behavior is always enabled:
  - The debugger does not apply heuristics.
  - The UI hides labels/distributions/pipeline unless the provider supplies them.
  - An informational banner indicates when no provider is detected and links to ODC docs.

## Error Handling

- Provider methods should never raise; return `None` to decline.
- The debugger handles exceptions defensively and degrades gracefully.

## Testing Guidance

- Ensure `policy_distribution()` never calls `select_action`.
- Ensure action names length equals `action_space.n`.
- Ensure determinism: same observation → same distribution.
- Provide small, dependency‑free examples in CI.
