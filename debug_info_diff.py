#!/usr/bin/env python3
"""Debug script to see what's different in info dicts."""

import sys

sys.path.insert(0, "src/backend")

from plume_nav_sim import PlumeSearchEnv

env = PlumeSearchEnv()

# First trajectory
env.reset(seed=123)
env.action_space.seed(123)
action = env.action_space.sample()
obs_0, rew_0, term_0, trunc_0, info_0 = env.step(action)

# Second trajectory
env.reset(seed=123)
env.action_space.seed(123)
action = env.action_space.sample()
obs_1, rew_1, term_1, trunc_1, info_1 = env.step(action)

print("INFO_0 keys:", sorted(info_0.keys()))
print("INFO_1 keys:", sorted(info_1.keys()))
print()

for key in sorted(set(info_0.keys()) | set(info_1.keys())):
    val0 = info_0.get(key, "<missing>")
    val1 = info_1.get(key, "<missing>")

    if val0 != val1:
        print(f"DIFF: {key}")
        print(f"  info_0: {val0}")
        print(f"  info_1: {val1}")
    else:
        print(f"SAME: {key} = {val0}")
