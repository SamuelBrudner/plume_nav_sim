from pathlib import Path


def test_core_no_fallbacks():
    content = Path("src/plume_nav_sim/core/__init__.py").read_text()
    assert "HYDRA_AVAILABLE" not in content
    assert "DictConfig = dict" not in content
