import pathlib

def test_dev_installation_section_mentions_script_and_alternative():
    text = pathlib.Path("docs/api_reference/index.md").read_text()
    assert "Unix-like systems" in text
    assert "./setup_env.sh --dev" in text
    assert "pip install -e .[dev]" in text
    assert "verbose logging" in text
    assert "fails fast" in text
