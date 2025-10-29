from __future__ import annotations

import numpy as np

import plume_nav_sim as pns
from plume_nav_sim.observations.history_wrappers import ConcentrationNBackWrapper


def test_concentration_nback_wrapper_smoke_with_real_env():
    # Build a real env with default concentration observations
    env = pns.make_env(
        grid_size=(32, 32),
        max_steps=20,
        action_type="run_tumble",
        observation_type="concentration",
        reward_type="step_penalty",
        render_mode=None,
    )
    try:
        wrapped = ConcentrationNBackWrapper(env, n=4)

        # Reset deterministically and check shape/space
        obs, _ = wrapped.reset(seed=123)
        assert wrapped.observation_space.contains(obs)
        assert obs.shape == (4,)

        # Step deterministically with a fixed action (RUN=0) to avoid RNG
        seq1 = [obs.copy()]
        for _ in range(3):
            obs, reward, term, trunc, info = wrapped.step(0)  # type: ignore[assignment]
            assert wrapped.observation_space.contains(obs)
            assert obs.shape == (4,)
            seq1.append(obs.copy())
            if term or trunc:
                break

        # Reset again with the same seed and replay the same fixed actions
        obs, _ = wrapped.reset(seed=123)
        seq2 = [obs.copy()]
        for _ in range(len(seq1) - 1):  # match length from first run
            obs, reward, term, trunc, info = wrapped.step(0)
            seq2.append(obs.copy())
            if term or trunc:
                break

        # The two sequences should match exactly under same seed and actions
        assert len(seq1) == len(seq2)
        for a, b in zip(seq1, seq2):
            np.testing.assert_allclose(a, b)
    finally:
        env.close()
