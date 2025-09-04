from pathlib import Path

def test_no_optional_dependency_flags():
    env_code = Path("src/plume_nav_sim/envs/plume_navigation_env.py").read_text()
    wind_code = Path("src/plume_nav_sim/models/wind/__init__.py").read_text()

    assert "GYMNASIUM_AVAILABLE" not in env_code, "GYMNASIUM_AVAILABLE flag should be removed"
    assert "HYDRA_AVAILABLE" not in wind_code, "HYDRA_AVAILABLE flag should be removed"
