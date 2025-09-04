from pathlib import Path

module_path = Path(__file__).resolve().parents[2] / 'src' / 'plume_nav_sim' / 'utils' / 'navigator_utils.py'
source = module_path.read_text()


def test_no_availability_flags():
    assert 'SEED_MANAGER_AVAILABLE' not in source
    assert 'ENHANCED_LOGGING_AVAILABLE' not in source
    assert 'HYDRA_AVAILABLE' not in source
