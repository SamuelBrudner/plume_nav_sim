from pathlib import Path
import re


def test_no_stdlib_logging_usage():
    project_root = Path(__file__).resolve().parents[1]
    source_dirs = [project_root / 'src', project_root / 'tests']
    pattern = re.compile(r'^\s*(import logging|from logging import)', re.MULTILINE)
    offenders = []
    for src_dir in source_dirs:
        for path in src_dir.rglob('*.py'):
            text = path.read_text()
            if pattern.search(text):
                offenders.append(str(path.relative_to(project_root)))
    assert not offenders, f'Standard logging imported in: {offenders}'
