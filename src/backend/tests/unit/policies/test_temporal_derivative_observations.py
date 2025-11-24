from __future__ import annotations

import numpy as np
import pytest

from plume_nav_sim.policies import (
    TemporalDerivativeDeterministicPolicy,
    TemporalDerivativePolicy,
)


def _make_policy(policy_cls, **overrides):
    if policy_cls is TemporalDerivativePolicy:
        base = {"eps": 0.0, "eps_after_turn": 0.0}
    else:
        base = {}
    base.update(overrides)
    return policy_cls(**base)


@pytest.mark.parametrize(
    "policy_cls",
    [TemporalDerivativePolicy, TemporalDerivativeDeterministicPolicy],
)
def test_accepts_dict_observation(policy_cls):
    policy = _make_policy(policy_cls)
    policy.reset(seed=0)

    obs1 = {"sensor_reading": np.array([0.1], dtype=np.float32)}
    obs2 = {"sensor_reading": np.array([0.2], dtype=np.float32)}

    assert policy.select_action(obs1, explore=False) == 0
    assert policy.select_action(obs2, explore=False) == 0


@pytest.mark.parametrize(
    "policy_cls",
    [TemporalDerivativePolicy, TemporalDerivativeDeterministicPolicy],
)
def test_tuple_observation_modality_index(policy_cls):
    policy = _make_policy(policy_cls)
    policy.reset(seed=123)

    obs1 = (
        np.array([0.05], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
    )
    obs2 = (
        np.array([0.07], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
    )

    policy.select_action(obs1, explore=False)
    assert policy.select_action(obs2, explore=False) == 0


@pytest.mark.parametrize(
    "policy_cls",
    [TemporalDerivativePolicy, TemporalDerivativeDeterministicPolicy],
)
def test_multi_sensor_array_requires_sensor_index(policy_cls):
    policy = _make_policy(policy_cls)
    policy.reset(seed=0)

    obs = np.array([0.1, 0.3], dtype=np.float32)
    with pytest.raises(ValueError, match="sensor_index"):
        policy.select_action(obs, explore=False)


@pytest.mark.parametrize(
    "policy_cls",
    [TemporalDerivativePolicy, TemporalDerivativeDeterministicPolicy],
)
def test_multi_sensor_array_with_sensor_index(policy_cls):
    policy = _make_policy(policy_cls, sensor_index=1)
    policy.reset(seed=99)

    obs1 = np.array([0.05, 0.25], dtype=np.float32)
    obs2 = np.array([0.10, 0.35], dtype=np.float32)

    policy.select_action(obs1, explore=False)
    assert policy.select_action(obs2, explore=False) == 0


def test_missing_concentration_key_errors():
    policy = TemporalDerivativePolicy(eps=0.0, eps_after_turn=0.0)
    policy.reset(seed=0)

    obs = {"not_concentration": np.array([0.1], dtype=np.float32)}
    with pytest.raises(ValueError, match="concentration"):
        policy.select_action(obs, explore=False)
