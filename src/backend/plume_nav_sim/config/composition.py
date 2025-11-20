"""
Unified composition API for plume_nav_sim.

This module centralizes composition-related types and helpers that were
previously spread across ``plume_nav_sim.compose``. It provides a stable
public surface for building environments and policies from simple specs
and for lightweight policy loading.

Contents
- PolicySpec and SimulationSpec: typed spec models
- load_policy/reset_policy_if_possible: dynamic policy loading helpers
- build_env/build_policy/prepare: high‑level composition helpers

Contributor guide
- Add new builtin policies by extending build_policy() handling
- Add new wrapper types by referencing them in SimulationSpec.observation_wrappers
- Prefer importing from plume_nav_sim.config.composition moving forward
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

import plume_nav_sim as pns
from plume_nav_sim.utils.spaces import is_space_subset

# ===== Specs =====

BuiltinPolicyName = Literal[
    "deterministic_td",
    "stochastic_td",
    "greedy_td",
    "random",
]


class PolicySpec(BaseModel):
    """Specification for constructing a policy.

    One of:
      - builtin: choose from curated names (deterministic_td, stochastic_td, random)
      - spec: dotted path ("module:Attr" or "module.Attr") to import

    kwargs are passed to builtin constructors and to class targets in dotted-path
    form. Callable targets (functions) are returned as-is.
    """

    model_config = ConfigDict(extra="forbid")

    builtin: Optional[BuiltinPolicyName] = Field(default=None)
    spec: Optional[str] = Field(default=None)
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("spec")
    @classmethod
    def _validate_spec(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        s = v.strip()
        if not s:
            raise ValueError("spec must be a non-empty string when provided")
        # lightweight shape check: must contain ':' or '.'
        if ":" not in s and "." not in s:
            raise ValueError(
                "spec must be a dotted path 'module:Attr' or 'module.Attr'"
            )
        return s

    @field_validator("kwargs")
    @classmethod
    def _validate_kwargs(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        # Shallow validation: keys must be str
        if not all(isinstance(k, str) for k in v):
            raise ValueError("kwargs keys must be strings")
        return v


class SimulationSpec(BaseModel):
    """Specification for constructing an environment and policy.

    Movie plume configuration
    -------------------------

    The ``plume`` and ``movie_*`` fields provide a lightweight front-end to
    the component-based environment factory used by :func:`prepare`.

    Under the movie metadata sidecar regime:

    - ``movie_path`` selects the recording to use.
    - ``movie_step_policy`` controls how environment steps map to frames.
    - ``movie_fps``, ``movie_pixel_to_grid``, ``movie_origin``,
      ``movie_extent`` and ``movie_h5_dataset`` are treated as validation
      aliases:

        * For raw media sources (files that will be ingested), these values
          are forwarded to :func:`plume_nav_sim.plume.movie_field.resolve_movie_dataset_path`
          and must agree with the movie metadata sidecar; they do not
          override sidecar metadata.
        * For already-ingested dataset directories (for example, ``*.zarr``),
          the same fields remain optional overrides for the dataset's own
          attrs and behave as before.

    Unit conventions
    ----------------

    - Time is measured in seconds: ``movie_fps`` is always interpreted as
      frames per second, and there is currently no separate ``time_unit``
      field.
    - For movie plumes, the physical spatial unit of the plume field is
      determined by the movie metadata sidecar's ``spatial_unit``; there is
      currently no separate ``SimulationSpec.spatial_unit`` override.
    """

    model_config = ConfigDict(extra="forbid")

    # Environment parameters (subset; defaults are taken from make_env if None)
    grid_size: Optional[Tuple[PositiveInt, PositiveInt]] = Field(default=None)
    source_location: Optional[Tuple[PositiveInt, PositiveInt]] = Field(default=None)
    start_location: Optional[Tuple[PositiveInt, PositiveInt]] = Field(default=None)
    goal_radius: Optional[float] = Field(default=None, ge=0.0)
    plume_sigma: Optional[float] = Field(default=None, ge=0.0)
    max_steps: Optional[PositiveInt] = Field(default=None)
    action_type: Optional[str] = Field(default=None)
    observation_type: Optional[str] = Field(default=None)
    reward_type: Optional[str] = Field(default=None)
    render: bool = Field(default=True)
    plume: Optional[Literal["static", "movie"]] = Field(default=None)
    movie_path: Optional[str] = Field(default=None)
    movie_fps: Optional[float] = Field(default=None, gt=0)
    movie_pixel_to_grid: Optional[Tuple[float, float]] = Field(default=None)
    movie_origin: Optional[Tuple[float, float]] = Field(default=None)
    movie_extent: Optional[Tuple[float, float]] = Field(default=None)
    movie_step_policy: Optional[Literal["wrap", "clamp"]] = Field(default=None)
    movie_h5_dataset: Optional[str] = Field(default=None)

    # Optional observation wrappers applied after env creation (in order).
    # Each wrapper is specified via dotted path and kwargs; the wrapper class
    # must have signature Wrapper(env, **kwargs) and return a gym.Env.
    class WrapperSpec(BaseModel):
        """Specification for an observation wrapper to apply in prepare().

        Fields
        ------
        spec
            Dotted path identifying the wrapper class ("module:Class" or "module.Class").
        kwargs
            Keyword arguments forwarded to the wrapper constructor.
        """

        model_config = ConfigDict(extra="forbid")
        spec: str
        kwargs: Dict[str, Any] = Field(default_factory=dict)

    observation_wrappers: List[WrapperSpec] = Field(default_factory=list)

    # Seeding / policy
    seed: Optional[int] = Field(default=123)
    policy: PolicySpec = Field(
        default_factory=lambda: PolicySpec(builtin="deterministic_td")
    )


# ===== Policy loading helpers =====


@dataclass
class LoadedPolicy:
    """Container for a loaded policy object and its spec string."""

    obj: Any
    spec: str


def _resolve_attr(mod: ModuleType, attr_path: str) -> Any:
    cur: Any = mod
    for part in attr_path.split("."):
        if not hasattr(cur, part):
            raise AttributeError(
                f"Attribute '{part}' not found while resolving '{attr_path}'"
            )
        cur = getattr(cur, part)
    return cur


def _import_module(module_name: str) -> ModuleType:
    return import_module(module_name)


def _import_longest_prefix(dotted: str) -> tuple[ModuleType, Optional[str]]:
    parts = dotted.split(".")
    # Try to import the longest importable prefix
    for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        try:
            mod = _import_module(mod_name)
            remainder = ".".join(parts[i:]) if i < len(parts) else None
            return mod, (remainder or None)
        except ModuleNotFoundError:
            continue
    # Fall back to plain import error
    raise ModuleNotFoundError(f"No importable module prefix found in '{dotted}'")


def load_policy(spec: str, *, kwargs: Optional[dict] = None) -> LoadedPolicy:
    """Load a policy from a spec string.

    Accepted forms:
    - "package.module:ClassOrCallable"
    - "package.module.ClassOrCallable" (last dot splits module vs. attribute)
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Policy spec must be a non-empty string")

    module_name: str
    attr_path: Optional[str]

    if ":" in spec:
        module_name, attr_path = spec.split(":", 1)
        if not attr_path:
            raise ValueError("Policy spec missing attribute after ':'")
        # Import the specified module directly for module:Attr form
        mod = _import_module(module_name)
    else:
        # For dotted form without ':', import the longest importable prefix as module
        # and treat the remainder as an attribute path.
        mod, attr_path = _import_longest_prefix(spec)

    target: Any
    target = _resolve_attr(mod, attr_path) if attr_path else mod
    # If target is a class, instantiate (passing kwargs if provided); otherwise return as-is
    try:
        if isinstance(target, type):
            if kwargs:
                if not isinstance(kwargs, dict):
                    raise TypeError(
                        "kwargs must be a dict when instantiating a class policy"
                    )
                obj = target(**kwargs)
            else:
                obj = target()
        else:
            obj = target
    except Exception as e:
        raise TypeError(f"Failed to instantiate policy '{spec}': {e}") from e

    return LoadedPolicy(obj=obj, spec=spec)


def reset_policy_if_possible(obj: Any, *, seed: Optional[int]) -> None:
    """Call `reset(seed=...)` on policy-like objects when available.

    Parameters
    ----------
    obj
        Policy object that may define a `reset(seed=...)` method.
    seed
        Seed to pass to the reset method.
    """
    with contextlib.suppress(Exception):
        obj.reset(seed=seed)  # type: ignore[attr-defined]


# ===== Builders =====


def _import_attr_for_wrapper(dotted: str) -> Any:
    """Import a class/object specified by a dotted or module:Attr path.

    Supports both "module:Attr.SubAttr" and "module.sub.Attr" forms.
    """
    if ":" in dotted:
        module_name, attr_path = dotted.split(":", 1)
        mod = import_module(module_name)
        target = mod
        for part in attr_path.split("."):
            target = getattr(target, part)
        return target
    parts = dotted.split(".")
    if len(parts) < 2:
        raise ValueError(f"Wrapper spec must include module and attribute: {dotted}")
    module_name = ".".join(parts[:-1])
    attr = parts[-1]
    mod = import_module(module_name)
    return getattr(mod, attr)


def _add_if_not_none(
    kwargs: dict[str, Any], key: str, value: Any, transform=None
) -> None:
    if value is None:
        return
    kwargs[key] = transform(value) if transform is not None else value


def _build_env_kwargs_from_spec(spec: SimulationSpec) -> dict[str, Any]:
    """Translate SimulationSpec into kwargs for pns.make_env.

    Only forwards parameters explicitly set on the spec so that
    make_env/env factories can apply their own defaults. For plume
    parameters, this helper maps ``plume_sigma`` onto the modern
    ``plume_params.sigma`` configuration expected by PlumeSearchEnv.
    """

    kwargs: dict[str, Any] = {}
    _add_if_not_none(kwargs, "grid_size", spec.grid_size, tuple)
    _add_if_not_none(kwargs, "source_location", spec.source_location, tuple)
    _add_if_not_none(kwargs, "start_location", spec.start_location, tuple)
    _add_if_not_none(kwargs, "goal_radius", spec.goal_radius, float)
    _add_if_not_none(kwargs, "max_steps", spec.max_steps, int)
    _add_if_not_none(kwargs, "action_type", spec.action_type)
    _add_if_not_none(kwargs, "observation_type", spec.observation_type)
    _add_if_not_none(kwargs, "reward_type", spec.reward_type)

    # Plume configuration: SimulationSpec exposes plume_sigma directly,
    # but the PlumeSearchEnv wrapper expects plume_params.sigma.
    if spec.plume_sigma is not None:
        kwargs["plume_params"] = {"sigma": float(spec.plume_sigma)}

    _add_if_not_none(kwargs, "plume", spec.plume)
    _add_if_not_none(kwargs, "movie_path", spec.movie_path)
    _add_if_not_none(kwargs, "movie_fps", spec.movie_fps, float)
    _add_if_not_none(
        kwargs,
        "movie_pixel_to_grid",
        spec.movie_pixel_to_grid,
        lambda v: tuple(v),
    )
    _add_if_not_none(
        kwargs,
        "movie_origin",
        spec.movie_origin,
        lambda v: tuple(v),
    )
    _add_if_not_none(
        kwargs,
        "movie_extent",
        spec.movie_extent,
        lambda v: tuple(v),
    )
    _add_if_not_none(kwargs, "movie_step_policy", spec.movie_step_policy)
    _add_if_not_none(kwargs, "movie_h5_dataset", spec.movie_h5_dataset)

    if spec.render:
        kwargs["render_mode"] = "rgb_array"

    return kwargs


def build_env(spec: SimulationSpec):
    """Construct an environment from SimulationSpec using pns.make_env.

    This is a thin adapter that converts the high-level SimulationSpec into
    the kwargs expected by the underlying PlumeSearchEnv/DI stack while
    leaving defaults to the factory.
    """
    kwargs = _build_env_kwargs_from_spec(spec)
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
        for w in sim.observation_wrappers:
            wrapper_cls = _import_attr_for_wrapper(w.spec)
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


__all__ = [
    # Specs
    "BuiltinPolicyName",
    "PolicySpec",
    "SimulationSpec",
    # Policy loading helpers
    "LoadedPolicy",
    "load_policy",
    "reset_policy_if_possible",
    # Builders
    "build_env",
    "build_policy",
    "prepare",
]
