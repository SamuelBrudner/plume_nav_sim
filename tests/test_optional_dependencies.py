import tomllib
from pathlib import Path


def test_seaborn_not_in_viz_optional_dependency():
    pyproject_text = Path('pyproject.toml').read_text()
    pyproject = tomllib.loads(pyproject_text)
    viz_deps = pyproject['project']['optional-dependencies']['viz']
    assert all('seaborn' not in dep for dep in viz_deps)
