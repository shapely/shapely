import pygeos
from pygeos.decorators import requires_geos, multithreading_enabled
from unittest import mock
import pytest
from itertools import chain
import sys

import numpy as np


@pytest.fixture
def mocked_geos_version():
    with mock.patch.object(pygeos.lib, "geos_version", new=(3, 7, 1)):
        yield "3.7.1"


def test_version():
    assert isinstance(pygeos.__version__, str)


def test_geos_version():
    expected = "{0:d}.{1:d}.{2:d}".format(*pygeos.geos_version)
    assert pygeos.geos_version_string == expected


@pytest.mark.skipif(
    sys.platform.startswith("win") and pygeos.geos_version[:2] == (3, 7),
    reason="GEOS_C_API_VERSION broken for GEOS 3.7.x on Windows",
)
def test_geos_capi_version():
    expected = "{0:d}.{1:d}.{2:d}-CAPI-{3:d}.{4:d}.{5:d}".format(
        *(pygeos.geos_version + pygeos.geos_capi_version)
    )
    assert pygeos.geos_capi_version_string == expected


@pytest.mark.parametrize("version", ["3.7.0", "3.7.1", "3.6.2"])
def test_requires_geos_ok(version, mocked_geos_version):
    @requires_geos(version)
    def foo():
        return "bar"

    assert foo() == "bar"


@pytest.mark.parametrize("version", ["3.7.2", "3.8.0", "3.8.1"])
def test_requires_geos_not_ok(version, mocked_geos_version):
    @requires_geos(version)
    def foo():
        return "bar"

    with pytest.raises(pygeos.UnsupportedGEOSOperation):
        foo()


@multithreading_enabled
def set_first_element(value, *args, **kwargs):
    for arg in chain(args, kwargs.values()):
        if hasattr(arg, "__setitem__"):
            arg[0] = value
            return arg


def test_multithreading_enabled_raises_arg():
    arr = np.empty((1,), dtype=object)

    # set_first_element cannot change the input array
    with pytest.raises(ValueError):
        set_first_element(42, arr)

    # afterwards, we can
    arr[0] = 42
    assert arr[0] == 42


def test_multithreading_enabled_raises_kwarg():
    arr = np.empty((1,), dtype=object)

    # set_first_element cannot change the input array
    with pytest.raises(ValueError):
        set_first_element(42, arr=arr)

    # writable flag goes to original state
    assert arr.flags.writeable


def test_multithreading_enabled_preserves_flag():
    arr = np.empty((1,), dtype=object)
    arr.flags.writeable = False

    # set_first_element cannot change the input array
    with pytest.raises(ValueError):
        set_first_element(42, arr)

    # writable flag goes to original state
    assert not arr.flags.writeable


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((np.empty((1,), dtype=float),), {}),  # float-dtype ndarray is untouched
        ((), {"a": np.empty((1,), dtype=float)}),
        (([1],), {}),  # non-ndarray is untouched
        ((), {"a": [1]}),
        ((), {"out": np.empty((1,), dtype=object)}),  # ufunc kwarg 'out' is untouched
        (
            (),
            {"where": np.empty((1,), dtype=object)},
        ),  # ufunc kwarg 'where' is untouched
    ],
)
def test_multithreading_enabled_ok(args, kwargs):
    result = set_first_element(42, *args, **kwargs)
    assert result[0] == 42
