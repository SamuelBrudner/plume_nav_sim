import os
import pytest

if os.environ.get("PLUME_DEBUGGER_GUI_TESTS") != "1":
    pytest.skip(
        "Debugger GUI tests require PLUME_DEBUGGER_GUI_TESTS=1",
        allow_module_level=True,
    )
