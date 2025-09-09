import importlib
import sys
import pytest


def test_import_from_source_succeeds():
    sys.modules.pop('plume_nav_sim', None)
    module = importlib.import_module('plume_nav_sim')
    assert hasattr(module, '__version__')
