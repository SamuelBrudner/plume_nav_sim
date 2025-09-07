import pathlib
import re


def test_no_unittest_usage_in_integration_tests():
    for path in pathlib.Path('tests/integration').glob('test_*.py'):
        source = path.read_text()
        assert re.search(r'\bimport unittest\b', source) is None
        assert 'unittest.mock' not in source
