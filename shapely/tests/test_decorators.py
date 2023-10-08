import ctypes
import os
import warnings

import pytest

from shapely.decorators import may_segfault


def my_unstable_func(event=None):
    if event == "segfault":
        ctypes.string_at(0)  # segfault
    elif event == "exit":
        exit(1)
    elif event == "raise":
        raise ValueError("This is a test")
    elif event == "warn":
        warnings.warn("This is a test", RuntimeWarning)
    elif event == "return":
        return "This is a test"


def test_may_segfault():
    if os.name == "nt":
        match = "access violation"
    else:
        match = r"Function crashed with exit code [-\d]+"
    with pytest.raises(OSError, match=match):
        may_segfault(my_unstable_func)("segfault")


def test_may_segfault_exit():
    with pytest.raises(OSError, match="Function crashed with exit code 1."):
        may_segfault(my_unstable_func)("exit")


def test_may_segfault_raises():
    with pytest.raises(ValueError, match="This is a test"):
        may_segfault(my_unstable_func)("raise")


def test_may_segfault_returns():
    assert may_segfault(my_unstable_func)("return") == "This is a test"


def test_may_segfault_warns():
    with pytest.warns(RuntimeWarning, match="This is a test"):
        may_segfault(my_unstable_func)("warn")
