from math import pi
from tempfile import TemporaryDirectory
from os.path import join

import pytest

from shapely.geometry import Point
from shapely.wkt import dumps, dump, load, loads


@pytest.fixture(scope="module")
def pipi():
    return Point((pi, -pi))


@pytest.fixture(scope="module")
def pipi4():
    return Point((pi*4, -pi*4))


@pytest.fixture(scope="module")
def null_geometry():
    return Point()


def test_wkt_simple(pipi):
    """.wkt and wkt.dumps() both do not trim by default."""
    assert pipi.wkt == "POINT ({0:.15f} {1:.15f})".format(pi, -pi)


def test_wkt(pipi4):
    """.wkt and wkt.dumps() both do not trim by default."""
    assert pipi4.wkt == "POINT ({0:.14f} {1:.14f})".format(pi*4, -pi*4)


def test_wkt_null(null_geometry):
    assert null_geometry.wkt == "GEOMETRYCOLLECTION EMPTY"


def test_dump_load(pipi4):
    with TemporaryDirectory() as directory_path:
        file = join(directory_path, "test.wkt")
        with open(file, "w") as file_pointer:
            dump(pipi4, file_pointer)
        with open(file, "r") as file_pointer:
            restored = load(file_pointer)

    assert pipi4 == restored


def test_dump_load_null_geometry(null_geometry):
    with TemporaryDirectory() as directory_path:
        file = join(directory_path, "test.wkt")
        with open(file, "w") as file_pointer:
            dump(null_geometry, file_pointer)
        with open(file, "r") as file_pointer:
            restored = load(file_pointer)

    # This is does not work with __eq__():
    assert null_geometry.equals(restored)


def test_dumps_loads(pipi4):
    assert dumps(pipi4) == "POINT ({0:.16f} {1:.16f})".format(pi*4, -pi*4)
    assert loads(dumps(pipi4)) == pipi4


def test_dumps_loads_null_geometry(null_geometry):
    assert dumps(null_geometry) == "GEOMETRYCOLLECTION EMPTY"
    # This is does not work with __eq__():
    assert loads(dumps(null_geometry)).equals(null_geometry)


def test_dumps_precision(pipi4):
    assert dumps(pipi4, rounding_precision=4) == "POINT ({0:.4f} {1:.4f})".format(pi*4, -pi*4)
