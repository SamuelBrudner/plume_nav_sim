import pathlib

def test_no_unittest_mock_usage_in_migration_test():
    source = pathlib.Path('tests/integration/test_v1_migration.py').read_text()
    assert 'from unittest import mock' not in source
