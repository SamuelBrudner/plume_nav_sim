from __future__ import annotations

from typing import Any, Optional, Tuple

import plume_nav_sim as pns
from plume_nav_sim.utils.spaces import is_space_subset

from .policy_loader import load_policy, reset_policy_if_possible
from .specs import PolicySpec, SimulationSpec


def build_env(spec: SimulationSpec):
    """Construct an environment from SimulationSpec using pns.make_env.

    Only forwards parameters explicitly set in the spec; others use defaults.
    """
    kwargs: dict[str, Any] = {}
    if spec.grid_size is not None:
        kwargs["grid_size"] = tuple(spec.grid_size)
    if spec.source_location is not None:
        kwargs["source_location"] = tuple(spec.source_location)
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
    if spec.plume is not None:
        kwargs["plume"] = spec.plume
    if spec.movie_path is not None:
        kwargs["movie_path"] = spec.movie_path
    if spec.movie_fps is not None:
        kwargs["movie_fps"] = float(spec.movie_fps)
    if spec.movie_pixel_to_grid is not None:
        kwargs["movie_pixel_to_grid"] = tuple(spec.movie_pixel_to_grid)
    if spec.movie_origin is not None:
        kwargs["movie_origin"] = tuple(spec.movie_origin)
    if spec.movie_extent is not None:
        kwargs["movie_extent"] = tuple(spec.movie_extent)
    if spec.movie_step_policy is not None:
        kwargs["movie_step_policy"] = spec.movie_step_policy
    kwargs["render_mode"] = "rgb_array" if spec.render else None

    return pns.make_env(**kwargs)


def _make_random_sampler(env) -> Any:
    class _Sampler:
        def __init__(self, env):
            self._env = env
            # Expose action_space so tests and runner can validate subset/identity
            self._space = getattr(env, "action_space", None)

        def __call__(self, _obs):
            return self._env.action_space.sample()

        @property
        def action_space(self):
            return self._space

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
            return _extracted_from_build_policy_23(policy_spec)
        # Note: run–tumble TD policy is intentionally not a builtin here.
        # External projects (or the demo package) should provide it via dotted-path.
        if name == "random":
            if env is None:
                raise ValueError("'random' builtin policy requires env to be provided")
            return _make_random_sampler(env)
        raise ValueError(f"Unknown builtin policy: {name}")

    if policy_spec.spec is None:
        raise ValueError("PolicySpec must declare either builtin or spec")

    loaded = load_policy(policy_spec.spec, kwargs=policy_spec.kwargs or None)
    return loaded.obj


# TODO Rename this here and in `build_policy`
def _extracted_from_build_policy_23(policy_spec):
    # Bacterial-like TD: deterministic forward on dC>=0,
    # uniform random among all actions when dC<=0 (diffusion)
    from plume_nav_sim.policies import TemporalDerivativePolicy

    params = dict(policy_spec.kwargs)
    params.setdefault("eps", 0.0)
    params.setdefault("eps_after_turn", 0.0)
    params.setdefault("eps_greedy_forward_bias", 0.0)
    # Uniform over all actions when non-increasing
    params["uniform_random_on_non_increase"] = True
    return TemporalDerivativePolicy(**params)


def prepare(sim: SimulationSpec) -> Tuple[Any, Any]:
    """Build (env, policy) pair from SimulationSpec.

    Applies deterministic reset(seed) to the policy if available.
    """
    env = build_env(sim)
    # Apply observation wrappers (ordered), specified via dotted-path classes
    # with signature Wrapper(env, **kwargs) -> gym.Env
    if getattr(sim, "observation_wrappers", None):
        from importlib import import_module

        def _import_attr(dotted: str) -> Any:
            if ":" in dotted:
                module_name, attr_path = dotted.split(":", 1)
                mod = import_module(module_name)
                target = mod
                for part in attr_path.split("."):
                    target = getattr(target, part)
                return target
            parts = dotted.split(".")
            if len(parts) < 2:
                raise ValueError(
                    f"Wrapper spec must include module and attribute: {dotted}"
                )
            module_name = ".".join(parts[:-1])
            attr = parts[-1]
            mod = import_module(module_name)
            return getattr(mod, attr)

        for w in sim.observation_wrappers:
            wrapper_cls = _import_attr(w.spec)
            env = wrapper_cls(env, **(w.kwargs or {}))

    policy = build_policy(sim.policy, env=env)
    # Composition-time subset check (structural, supports common spaces)
    env_space = getattr(env, "action_space", None)
    pol_space = getattr(policy, "action_space", None)
    if (
        env_space is not None
        and pol_space is not None
        and not is_space_subset(pol_space, env_space)
    ):
        raise ValueError(
            "Policy action space must be a subset of the environment's action space"
        )
    reset_policy_if_possible(policy, seed=sim.seed)
    return env, policy
