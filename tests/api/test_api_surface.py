import importlib

def test_api_surface_is_clean():
    api = importlib.import_module("odor_plume_nav.api")
    public_api = set(dir(api))
    intended = {
        "create_navigator",
        "create_video_plume",
        "run_plume_simulation",
        "visualize_simulation_results",
    }
    forbidden = {
        "_merge_config_with_args",
        "_validate_positions",
        "_load_config",
        "_load_navigator_from_config",
    }
    assert intended.issubset(public_api), f"Missing public API: {intended - public_api}"
    assert forbidden.isdisjoint(public_api), f"Private helpers exposed: {forbidden & public_api}"
