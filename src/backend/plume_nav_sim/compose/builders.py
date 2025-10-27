from __future__ import annotations

from typing import Any, Optional, Tuple

import plume_nav_sim as pns

from .policy_loader import load_policy, reset_policy_if_possible
from .specs import PolicySpec, SimulationSpec


def build_env(spec: SimulationSpec):
    """Construct an environment from SimulationSpec using pns.make_env.

    Only forwards parameters explicitly set in the spec; others use defaults.
    """
    kwargs: dict[str, Any] = {}
    if spec.grid_size is not None:
        kwargs["grid_size"] = tuple(spec.grid_size)
    if spec.start_location is not None:
        kwargs["start_location"] = tuple(spec.start_location)
    if spec.goal_radius is not None:
        kwargs["goal_radius"] = float(spec.goal_radius)
    if spec.plume_sigma is not None:
        kwargs["plume_sigma"] = float(spec.plume_sigma)
    if spec.max_steps is not None:
        kwargs["max_steps"] = int(spec.max_steps)
    if spec.action_type is not None:
        kwargs["action_type"] = spec.action_type
    if spec.observation_type is not None:
        kwargs["observation_type"] = spec.observation_type
    if spec.reward_type is not None:
        kwargs["reward_type"] = spec.reward_type
    kwargs["render_mode"] = "rgb_array" if spec.render else None

    env = pns.make_env(**kwargs)
    return env


def _make_random_sampler(env) -> Any:
    class _Sampler:
        def __init__(self, env):
            self._env = env

        def __call__(self, _obs):
            return self._env.action_space.sample()

    return _Sampler(env)


def build_policy(policy_spec: PolicySpec, *, env: Optional[Any] = None) -> Any:
    """Construct a policy object from PolicySpec.

    - builtin policies are instantiated directly
    - dotted-path spec imports and instantiates class targets with kwargs if provided
    - function targets are returned as-is

    The 'random' builtin requires access to env; if not provided, raises.
    """
    if policy_spec.builtin:
        name = policy_spec.builtin
        if name == "deterministic_td":
            from plume_nav_sim.policies import TemporalDerivativeDeterministicPolicy

            return TemporalDerivativeDeterministicPolicy(**policy_spec.kwargs)
        if name == "stochastic_td":
            from plume_nav_sim.policies import TemporalDerivativePolicy

            return TemporalDerivativePolicy(**policy_spec.kwargs)
        if name == "greedy_td":
            from plume_nav_sim.policies import TemporalDerivativeDeterministicPolicy

            params = dict(policy_spec.kwargs)
            params.setdefault("alternate_cast", False)
            return TemporalDerivativeDeterministicPolicy(**params)
        if name == "random":
            if env is None:
                raise ValueError("'random' builtin policy requires env to be provided")
            return _make_random_sampler(env)
        raise ValueError(f"Unknown builtin policy: {name}")

    if policy_spec.spec is None:
        raise ValueError("PolicySpec must declare either builtin or spec")

    loaded = load_policy(policy_spec.spec, kwargs=policy_spec.kwargs or None)
    return loaded.obj


def prepare(sim: SimulationSpec) -> Tuple[Any, Any]:
    """Build (env, policy) pair from SimulationSpec.

    Applies deterministic reset(seed) to the policy if available.
    """
    env = build_env(sim)
    policy = build_policy(sim.policy, env=env)
    reset_policy_if_possible(policy, seed=sim.seed)
    return env, policy
