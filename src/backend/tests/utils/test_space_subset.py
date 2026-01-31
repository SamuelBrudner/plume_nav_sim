from __future__ import annotations

import numpy as np
from gymnasium import spaces

from plume_nav_sim._compat import is_space_subset


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


def test_is_space_subset_multidiscrete_and_multibinary():
    md_pol = spaces.MultiDiscrete([2, 3])
    md_env = spaces.MultiDiscrete([2, 5])
    assert is_space_subset(md_pol, md_env)
    assert not is_space_subset(md_env, md_pol)

    mb_pol = spaces.MultiBinary(3)
    mb_env = spaces.MultiBinary(5)
    assert is_space_subset(mb_pol, mb_env)
    assert not is_space_subset(mb_env, mb_pol)


def test_is_space_subset_mismatched_types():
    box_space = spaces.Box(low=0, high=1, shape=(1,))
    assert not is_space_subset(spaces.Discrete(2), box_space)

    tuple_space = spaces.Tuple((spaces.Discrete(1),))
    dict_space = spaces.Dict({"a": spaces.Discrete(1)})
    assert not is_space_subset(tuple_space, dict_space)
