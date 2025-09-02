import subprocess
import sys
import textwrap


def run_in_subprocess(code: str):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_import_fails_without_navigator():
    code = textwrap.dedent(
        """
        import os, sys, types
        sys.path.insert(0, os.path.abspath('src'))
        # stub plume_nav_sim package with empty core
        pkg = types.ModuleType('plume_nav_sim')
        pkg.__path__ = [os.path.abspath('src/plume_nav_sim')]
        core_stub = types.ModuleType('plume_nav_sim.core')
        sys.modules['plume_nav_sim'] = pkg
        sys.modules['plume_nav_sim.core'] = core_stub
        import plume_nav_sim.api.navigation
        """
    )
    result = run_in_subprocess(code)
    assert result.returncode != 0
    assert 'ImportError' in result.stderr or 'ModuleNotFoundError' in result.stderr


def test_create_env_fails_without_gymnasium():
    code = textwrap.dedent(
        """
        import os, sys, builtins
        sys.path.insert(0, os.path.abspath('src'))
        real_import = builtins.__import__
        def fake_import(name, *args, **kwargs):
            if name == 'gymnasium':
                raise ImportError('gymnasium missing')
            return real_import(name, *args, **kwargs)
        builtins.__import__ = fake_import
        import plume_nav_sim.api.navigation as navigation
        navigation.create_gymnasium_environment()
        """
    )
    result = run_in_subprocess(code)
    assert result.returncode != 0
    assert 'ImportError' in result.stderr
