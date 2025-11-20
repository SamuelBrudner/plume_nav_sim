from __future__ import annotations

import numpy as np
from gymnasium import spaces

from plume_nav_sim.utils.spaces import is_space_subset


def test_is_space_subset_discrete():
    assert is_space_subset(spaces.Discrete(2), spaces.Discrete(3))
    assert is_space_subset(spaces.Discrete(3), spaces.Discrete(3))
    assert not is_space_subset(spaces.Discrete(4), spaces.Discrete(3))


def test_is_space_subset_box():
    p = spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)
    e = spaces.Box(low=np.array([0.0]), high=np.array([2.0]), dtype=np.float32)
    assert is_space_subset(p, e)
    assert not is_space_subset(e, p)
    # Shape mismatch
    p2 = spaces.Box(low=np.zeros((2,)), high=np.ones((2,)), dtype=np.float32)
    assert not is_space_subset(p2, e)


def test_is_space_subset_tuple_and_dict():
    t_pol = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(1)))
    t_env = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(1)))
    assert is_space_subset(t_pol, t_env)
    assert not is_space_subset(t_env, t_pol)

    d_pol = spaces.Dict({"a": spaces.Discrete(2)})
    d_env = spaces.Dict({"a": spaces.Discrete(3), "b": spaces.Discrete(1)})
    assert is_space_subset(d_pol, d_env)
    assert not is_space_subset(d_env, d_pol)
