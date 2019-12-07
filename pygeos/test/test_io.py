import numpy as np
import pygeos
import pytest
from unittest import mock

from .common import all_types, point


POINT11_WKB = (
    b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?"
)


class ShapelyGeometryMock:
    def __init__(self, g):
        self.g = g
        self.__geom__ = g._ptr if hasattr(g, "_ptr") else g

    def __array_interface__(self):
        # this should not be called
        raise NotImplementedError()


def test_from_wkt():
    expected = pygeos.points(1, 1)
    actual = pygeos.from_wkt("POINT (1 1)")
    assert pygeos.equals(actual, expected)
    # also accept bytes
    actual = pygeos.from_wkt(b"POINT (1 1)")
    assert pygeos.equals(actual, expected)


def test_from_wkt_none():
    # None propagates
    assert pygeos.from_wkt(None) is None


def test_from_wkt_exceptions():
    with pytest.raises(TypeError, match="Expected bytes, got int"):
        pygeos.from_wkt(1)

    with pytest.raises(pygeos.GEOSException):
        pygeos.from_wkt("")

    with pytest.raises(pygeos.GEOSException):
        pygeos.from_wkt("NOT A WKT STRING")


@pytest.mark.parametrize("geom", all_types)
def test_from_wkt_all_types(geom):
    wkt = pygeos.to_wkt(geom)
    actual = pygeos.from_wkt(wkt)
    assert pygeos.equals(actual, geom)


@pytest.mark.parametrize(
    "wkt",
    ("POINT EMPTY", "LINESTRING EMPTY", "POLYGON EMPTY", "GEOMETRYCOLLECTION EMPTY"),
)
def test_from_wkt_empty(wkt):
    geom = pygeos.from_wkt(wkt)
    assert pygeos.is_geometry(geom).all()
    assert pygeos.is_empty(geom).all()
    assert pygeos.to_wkt(geom) == wkt


def test_from_wkb():
    expected = pygeos.points(1, 1)
    actual = pygeos.from_wkb(POINT11_WKB)
    assert pygeos.equals(actual, expected)


def test_from_wkb_hex():
    # HEX form
    expected = pygeos.points(1, 1)
    actual = pygeos.from_wkb("0101000000000000000000F03F000000000000F03F")
    assert pygeos.equals(actual, expected)
    actual = pygeos.from_wkb(b"0101000000000000000000F03F000000000000F03F")
    assert pygeos.equals(actual, expected)


def test_from_wkb_none():
    # None propagates
    assert pygeos.from_wkb(None) is None


def test_from_wkb_exceptions():
    with pytest.raises(TypeError, match="Expected bytes, got int"):
        pygeos.from_wkb(1)

    with pytest.raises(pygeos.GEOSException):
        pygeos.from_wkb(b"\x01\x01\x00\x00\x00\x00")


@pytest.mark.parametrize("geom", all_types)
@pytest.mark.parametrize("use_hex", [False, True])
@pytest.mark.parametrize("byte_order", [0, 1])
def test_from_wkb_all_types(geom, use_hex, byte_order):
    wkb = pygeos.to_wkb(geom, hex=use_hex, byte_order=byte_order)
    actual = pygeos.from_wkb(wkb)
    assert pygeos.equals(actual, geom)


@pytest.mark.parametrize(
    "wkt", ("LINESTRING EMPTY", "POLYGON EMPTY", "GEOMETRYCOLLECTION EMPTY")
)
def test_from_wkb_empty(wkt):
    wkb = pygeos.to_wkb(pygeos.from_wkt(wkt))
    geom = pygeos.from_wkb(wkb)
    assert pygeos.is_geometry(geom).all()
    assert pygeos.is_empty(geom).all()
    assert pygeos.to_wkb(geom) == wkb


def test_from_wkb_empty_point():
    geom = pygeos.from_wkt("POINT EMPTY")
    with pytest.raises(pygeos.GEOSException):
        pygeos.to_wkb(geom)


def test_to_wkt():
    point = pygeos.points(1, 1)
    actual = pygeos.to_wkt(point)
    assert actual == "POINT (1 1)"

    actual = pygeos.to_wkt(point, trim=False)
    assert actual == "POINT (1.000000 1.000000)"

    actual = pygeos.to_wkt(point, rounding_precision=3, trim=False)
    assert actual == "POINT (1.000 1.000)"


def test_to_wkt_3D():
    # 3D points
    point_z = pygeos.points(1, 1, 1)
    actual = pygeos.to_wkt(point_z)
    assert actual == "POINT Z (1 1 1)"
    actual = pygeos.to_wkt(point_z, output_dimension=3)
    assert actual == "POINT Z (1 1 1)"

    actual = pygeos.to_wkt(point_z, output_dimension=2)
    assert actual == "POINT (1 1)"

    actual = pygeos.to_wkt(point_z, old_3d=True)
    assert actual == "POINT (1 1 1)"


def test_to_wkt_none():
    # None propagates
    assert pygeos.to_wkt(None) is None


def test_to_wkt_exceptions():
    with pytest.raises(TypeError):
        pygeos.to_wkt(1)

    with pytest.raises(pygeos.GEOSException):
        pygeos.to_wkt(point, output_dimension=4)


def test_to_wkb():
    point = pygeos.points(1, 1)
    actual = pygeos.to_wkb(point)
    assert actual == POINT11_WKB


def test_to_wkb_hex():
    point = pygeos.points(1, 1)
    actual = pygeos.to_wkb(point, hex=True)
    le = "01"
    point_type = "01000000"
    coord = "000000000000F03F"  # 1.0 as double (LE)
    assert actual == le + point_type + 2 * coord


def test_to_wkb_3D():
    point_z = pygeos.points(1, 1, 1)
    actual = pygeos.to_wkb(point_z)
    # fmt: off
    assert actual == b"\x01\x01\x00\x00\x80\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?"  # noqa
    # fmt: on
    actual = pygeos.to_wkb(point_z, output_dimension=2)
    assert actual == POINT11_WKB


def test_to_wkb_none():
    # None propagates
    assert pygeos.to_wkb(None) is None


def test_to_wkb_exceptions():
    with pytest.raises(TypeError):
        pygeos.to_wkb(1)

    with pytest.raises(pygeos.GEOSException):
        pygeos.to_wkb(point, output_dimension=4)


def test_to_wkb_byte_order():
    point = pygeos.points(1.0, 1.0)
    be = b"\x00"
    le = b"\x01"
    point_type = b"\x01\x00\x00\x00"  # 1 as 32-bit uint (LE)
    coord = b"\x00\x00\x00\x00\x00\x00\xf0?"  # 1.0 as double (LE)

    assert pygeos.to_wkb(point, byte_order=1) == le + point_type + 2 * coord
    assert pygeos.to_wkb(point, byte_order=0) == be + point_type[::-1] + 2 * coord[::-1]


def test_to_wkb_srid():
    # hex representation of POINT (0 0) with SRID=4
    ewkb = "01010000200400000000000000000000000000000000000000"
    wkb = "010100000000000000000000000000000000000000"

    actual = pygeos.from_wkb(ewkb)
    assert pygeos.to_wkt(actual, trim=True) == "POINT (0 0)"

    assert pygeos.to_wkb(actual, hex=True) == wkb
    assert pygeos.to_wkb(actual, hex=True, include_srid=True) == ewkb

    point = pygeos.points(1, 1)
    point_with_srid = pygeos.set_srid(point, np.int32(4326))
    result = pygeos.to_wkb(point_with_srid, include_srid=True)
    assert np.frombuffer(result[5:9], "<u4").item() == 4326


@pytest.mark.parametrize("geom", all_types)
@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.shapely_geos_version", pygeos.geos_capi_version_string)
def test_from_shapely(geom):
    actual = pygeos.from_shapely(ShapelyGeometryMock(geom))
    assert isinstance(geom, pygeos.Geometry)
    assert pygeos.equals(geom, actual)
    assert geom._ptr != actual._ptr


@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.shapely_geos_version", pygeos.geos_capi_version_string)
def test_from_shapely_arr():
    actual = pygeos.from_shapely([ShapelyGeometryMock(point), None])
    assert pygeos.equals(point, actual[0])


@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.shapely_geos_version", pygeos.geos_capi_version_string)
def test_from_shapely_none():
    actual = pygeos.from_shapely(None)
    assert actual is None


@pytest.mark.parametrize("geom", [1, 2.3, "x", ShapelyGeometryMock(None)])
@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.shapely_geos_version", pygeos.geos_capi_version_string)
def test_from_shapely_error(geom):
    with pytest.raises(TypeError):
        pygeos.from_shapely(geom)


# We have >= 3.5 in PyGEOS. Test with some random older version.
@mock.patch("pygeos.io.shapely_geos_version", "2.3.4-abc")
def test_from_shapely_incompatible_versions():
    with pytest.raises(ImportError):
        pygeos.from_shapely(point)
