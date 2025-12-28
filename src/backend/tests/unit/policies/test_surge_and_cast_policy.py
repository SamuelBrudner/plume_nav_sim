from plume_nav_sim.policies import SurgeAndCastPolicy


def test_surges_on_positive_gradient():
    policy = SurgeAndCastPolicy(threshold=0.0, persistence=0)
    policy.reset(seed=0)

    assert policy.select_action(0.1, explore=False) == 0
    assert policy.select_action(0.2, explore=False) == 0


def test_casts_then_persists_and_alternates():
    policy = SurgeAndCastPolicy(
        threshold=0.0, cast_mode="alternating_turn", persistence=2
    )
    policy.reset(seed=0)

    assert policy.select_action(0.2, explore=False) == 0
    assert policy.select_action(0.1, explore=False) == 1
    assert policy.select_action(0.05, explore=False) == 0  # persistence step 1
    assert policy.select_action(0.04, explore=False) == 0  # persistence step 2
    assert policy.select_action(0.03, explore=False) == 2  # alternate turn


def test_random_cast_mode_respects_explore_flag():
    policy = SurgeAndCastPolicy(threshold=0.0, cast_mode="random_turn", persistence=0)
    policy.reset(seed=0)

    policy.select_action(0.2, explore=False)
    assert (
        policy.select_action(0.1, explore=False) == 1
    )  # deterministic when explore=False

    policy.reset(seed=123)
    policy.select_action(0.2, explore=True)
    random_action = policy.select_action(0.0, explore=True)
    assert random_action in (1, 2)


def test_zigzag_persists_direction_across_positive_gradients():
    policy = SurgeAndCastPolicy(
        threshold=0.0, cast_mode="zigzag", persistence=0, eps_seed=7
    )
    policy.reset(seed=0)

    policy.select_action(0.3, explore=False)
    assert policy.select_action(0.1, explore=False) == 1
    assert policy.select_action(0.2, explore=False) == 0  # surge on recovery
    assert policy.select_action(0.05, explore=False) == 2  # flip direction after surge
