import pygeos
from pygeos.geos import requires_geos
from unittest import mock
import pytest


@pytest.fixture
def mocked_geos_version():
    with mock.patch.object(pygeos.lib, "geos_version", new=(3, 7, 1)):
        yield "3.7.1"


def test_version():
    assert isinstance(pygeos.__version__, str)


def test_geos_version():
    expected = "{0:d}.{1:d}.{2:d}".format(*pygeos.geos_version)
    assert pygeos.geos_version_string == expected


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
