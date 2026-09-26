"""Path setup, build guard, and a minimal coroutine-test runner.

`pytest-asyncio` is not installed in the shared test environment, so
`pytest_pyfunc_call` runs coroutine test functions with `asyncio.run` instead.
Every `async def test_*` in this suite is executed for real; nothing is skipped.
"""

import asyncio
import inspect
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-aiosqlite.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-aiosqlite.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    test_fn = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_fn):
        return None
    kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    asyncio.run(test_fn(**kwargs))
    return True
