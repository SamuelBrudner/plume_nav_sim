import subprocess
import sys
import textwrap


def run_in_subprocess(code: str):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_import_fails_without_gymnasium():
    code = textwrap.dedent(
        """
        import builtins, sys, os
        sys.path.insert(0, os.path.abspath('src'))
        real_import = builtins.__import__
        def fake_import(name, *args, **kwargs):
            if name == 'odor_plume_nav.environments.gymnasium_env':
                raise ImportError('gymnasium env missing')
            return real_import(name, *args, **kwargs)
        builtins.__import__ = fake_import
        import odor_plume_nav.api.navigation
        """
    )
    result = run_in_subprocess(code)
    assert result.returncode != 0
    assert 'ImportError' in result.stderr


def test_import_fails_without_frame_cache():
    code = textwrap.dedent(
        """
        import builtins, sys, os
        sys.path.insert(0, os.path.abspath('src'))
        real_import = builtins.__import__
        def fake_import(name, *args, **kwargs):
            if name == 'odor_plume_nav.cache.frame_cache':
                raise ImportError('frame cache missing')
            return real_import(name, *args, **kwargs)
        builtins.__import__ = fake_import
        import odor_plume_nav.api.navigation
        """
    )
    result = run_in_subprocess(code)
    assert result.returncode != 0
    assert 'ImportError' in result.stderr
