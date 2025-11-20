"""Composition utilities for building and configuring simulations.

This package is UI-agnostic and focuses on converting specs/configs into
runtime objects (envs, policies, callbacks). Keep runner concerns separate.
"""

from .builders import build_env, build_policy, prepare  # noqa: F401
from .policy_loader import (  # noqa: F401
    LoadedPolicy,
    load_policy,
    reset_policy_if_possible,
)
from .specs import PolicySpec  # noqa: F401
from .specs import SimulationSpec  # noqa: F401

# Re-export nested WrapperSpec for convenience
WrapperSpec = SimulationSpec.WrapperSpec  # noqa: F401
