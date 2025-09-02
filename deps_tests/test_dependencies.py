import tomllib
from pathlib import Path


def test_requirements_sync():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    base = set(pyproject["project"]["dependencies"])
    dev = set(pyproject["project"]["optional-dependencies"]["dev"])
    viz = set(pyproject["project"]["optional-dependencies"]["viz"])

    req_base = set(Path("requirements.txt").read_text().splitlines())
    req_dev = set(Path("requirements-dev.txt").read_text().splitlines())
    req_viz = set(Path("requirements-viz.txt").read_text().splitlines())

    assert req_base == base, "requirements.txt out of sync with pyproject dependencies"
    assert req_dev == base | dev, "requirements-dev.txt out of sync with pyproject dev dependencies"
    assert req_viz == base | viz, "requirements-viz.txt out of sync with pyproject viz dependencies"
