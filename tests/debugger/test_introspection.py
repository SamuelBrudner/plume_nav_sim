from plume_nav_debugger.inspector.introspection import (
    format_pipeline,
    get_env_chain_names,
)


class _Inner:
    pass


class _Wrapper:
    def __init__(self, env):
        self.env = env


class _Top:
    def __init__(self):
        self.env = _Wrapper(_Wrapper(_Inner()))


def test_get_env_chain_names_simple_chain():
    root = _Top()
    names = get_env_chain_names(root)
    assert names[0] == "_Top"
    assert "_Inner" in names
    s = format_pipeline(names)
    assert "->" in s
