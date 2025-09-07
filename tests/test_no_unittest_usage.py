import pathlib
import re
from itertools import chain


def test_no_unittest_usage_in_selected_tests():
    paths = chain(
        pathlib.Path("tests/integration").glob("test_*.py"),
        pathlib.Path("tests/examples").glob("test_*.py"),
        [pathlib.Path("tests/test_logging_json.py")],
    )
    for path in paths:
        source = path.read_text()
        assert re.search(r"\bimport unittest\b", source) is None
        assert "unittest.mock" not in source

