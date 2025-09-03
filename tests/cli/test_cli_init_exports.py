import pathlib


def test_cli_init_has_no_is_available_definition():
    contents = pathlib.Path('src/plume_nav_sim/cli/__init__.py').read_text()
    assert 'def is_available' not in contents
