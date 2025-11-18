from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

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
    """Specification for constructing an environment and policy."""

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
