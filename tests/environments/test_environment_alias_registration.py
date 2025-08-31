import gymnasium as gym


def test_snake_case_environment_id_registered():
    import plume_nav_sim.envs  # Trigger environment registration
    assert 'PlumeNavSim-v0' in gym.envs.registry
    assert 'plume_nav_sim_v0' in gym.envs.registry
    spec_main = gym.envs.registry['PlumeNavSim-v0']
    spec_alias = gym.envs.registry['plume_nav_sim_v0']
    assert spec_main.entry_point == spec_alias.entry_point
    assert spec_main.kwargs == spec_alias.kwargs
