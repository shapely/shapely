import pickle
import struct
from unittest import mock

import numpy as np
import pytest

import pygeos

from .common import all_types, empty_point, point, point_z

# fmt: off
POINT11_WKB = b"\x01\x01\x00\x00\x00" + struct.pack("<2d", 1.0, 1.0)
NAN = struct.pack("<d", float("nan"))
POINT_NAN_WKB = b'\x01\x01\x00\x00\x00' + (NAN * 2)
POINTZ_NAN_WKB = b'\x01\x01\x00\x00\x80' + (NAN * 3)
MULTIPOINT_NAN_WKB = b'\x01\x04\x00\x00\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00' + (NAN * 2)
MULTIPOINTZ_NAN_WKB = b'\x01\x04\x00\x00\x80\x01\x00\x00\x00\x01\x01\x00\x00\x80' + (NAN * 3)
GEOMETRYCOLLECTION_NAN_WKB = b'\x01\x07\x00\x00\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00' + (NAN * 2)
GEOMETRYCOLLECTIONZ_NAN_WKB = b'\x01\x07\x00\x00\x80\x01\x00\x00\x00\x01\x01\x00\x00\x80' + (NAN * 3)
NESTED_COLLECTION_NAN_WKB = b'\x01\x07\x00\x00\x00\x01\x00\x00\x00\x01\x04\x00\x00\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00' + (NAN * 2)
NESTED_COLLECTIONZ_NAN_WKB = b'\x01\x07\x00\x00\x80\x01\x00\x00\x00\x01\x04\x00\x00\x80\x01\x00\x00\x00\x01\x01\x00\x00\x80' + (NAN * 3)
# fmt: on


class ShapelyGeometryMock:
    def __init__(self, g):
        self.g = g
        self.__geom__ = g._ptr if hasattr(g, "_ptr") else g

    @property
    def __array_interface__(self):
        # this should not be called
        # (starting with numpy 1.20 it is called, but not used)
        return np.array([1.0, 2.0]).__array_interface__

    @property
    def wkb(self):
        return pygeos.to_wkb(self.g)

    @property
    def geom_type(self):
        idx = pygeos.get_type_id(self.g)
        return [
            "None",
            "Point",
            "LineString",
            "LinearRing",
            "Polygon",
            "MultiPoint",
            "MultiLineString",
            "MultiPolygon",
            "GeometryCollection",
        ][idx]

    @property
    def is_empty(self):
        return pygeos.is_empty(self.g)


class ShapelyPreparedMock:
    def __init__(self, g):
        self.context = ShapelyGeometryMock(g)


def shapely_wkb_loads_mock(wkb):
    geom = pygeos.from_wkb(wkb)
    return ShapelyGeometryMock(geom)


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

    with pytest.raises(
        pygeos.GEOSException, match="Expected word but encountered end of stream"
    ):
        pygeos.from_wkt("")

    with pytest.raises(pygeos.GEOSException, match="Unknown type: 'NOT'"):
        pygeos.from_wkt("NOT A WKT STRING")


def test_from_wkt_warn_on_invalid():
    with pytest.warns(Warning, match="Invalid WKT"):
        pygeos.from_wkt("", on_invalid="warn")

    with pytest.warns(Warning, match="Invalid WKT"):
        pygeos.from_wkt("NOT A WKT STRING", on_invalid="warn")


def test_from_wkb_ignore_on_invalid():
    with pytest.warns(None):
        pygeos.from_wkt("", on_invalid="ignore")

    with pytest.warns(None):
        pygeos.from_wkt("NOT A WKT STRING", on_invalid="ignore")


def test_from_wkt_on_invalid_unsupported_option():
    with pytest.raises(ValueError, match="not a valid option"):
        pygeos.from_wkt(b"\x01\x01\x00\x00\x00\x00", on_invalid="unsupported_option")


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

    # invalid WKB
    with pytest.raises(pygeos.GEOSException, match="Unexpected EOF parsing WKB"):
        result = pygeos.from_wkb(b"\x01\x01\x00\x00\x00\x00")
        assert result is None

    # invalid ring in WKB
    with pytest.raises(
        pygeos.GEOSException,
        match="Invalid number of points in LinearRing found 3 - must be 0 or >= 4",
    ):
        result = pygeos.from_wkb(
            b"\x01\x03\x00\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00P}\xae\xc6\x00\xb15A\x00\xde\x02I\x8e^=A0n\xa3!\xfc\xb05A\xa0\x11\xa5=\x90^=AP}\xae\xc6\x00\xb15A\x00\xde\x02I\x8e^=A"
        )
        assert result is None


def test_from_wkb_warn_on_invalid_warn():
    # invalid WKB
    with pytest.warns(Warning, match="Invalid WKB"):
        result = pygeos.from_wkb(b"\x01\x01\x00\x00\x00\x00", on_invalid="warn")
        assert result is None

    # invalid ring in WKB
    with pytest.warns(Warning, match="Invalid WKB"):
        result = pygeos.from_wkb(
            b"\x01\x03\x00\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00P}\xae\xc6\x00\xb15A\x00\xde\x02I\x8e^=A0n\xa3!\xfc\xb05A\xa0\x11\xa5=\x90^=AP}\xae\xc6\x00\xb15A\x00\xde\x02I\x8e^=A",
            on_invalid="warn",
        )
        assert result is None


def test_from_wkb_ignore_on_invalid_ignore():
    # invalid WKB
    with pytest.warns(None) as w:
        result = pygeos.from_wkb(b"\x01\x01\x00\x00\x00\x00", on_invalid="ignore")
        assert result is None
        assert len(w) == 0  # no warning

    # invalid ring in WKB
    with pytest.warns(None) as w:
        result = pygeos.from_wkb(
            b"\x01\x03\x00\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00P}\xae\xc6\x00\xb15A\x00\xde\x02I\x8e^=A0n\xa3!\xfc\xb05A\xa0\x11\xa5=\x90^=AP}\xae\xc6\x00\xb15A\x00\xde\x02I\x8e^=A",
            on_invalid="ignore",
        )
        assert result is None
        assert len(w) == 0  # no warning


def test_from_wkb_on_invalid_unsupported_option():
    with pytest.raises(ValueError, match="not a valid option"):
        pygeos.from_wkb(b"\x01\x01\x00\x00\x00\x00", on_invalid="unsupported_option")


@pytest.mark.parametrize("geom", all_types)
@pytest.mark.parametrize("use_hex", [False, True])
@pytest.mark.parametrize("byte_order", [0, 1])
def test_from_wkb_all_types(geom, use_hex, byte_order):
    wkb = pygeos.to_wkb(geom, hex=use_hex, byte_order=byte_order)
    actual = pygeos.from_wkb(wkb)
    assert pygeos.equals(actual, geom)


@pytest.mark.parametrize(
    "wkt",
    ("POINT EMPTY", "LINESTRING EMPTY", "POLYGON EMPTY", "GEOMETRYCOLLECTION EMPTY"),
)
def test_from_wkb_empty(wkt):
    wkb = pygeos.to_wkb(pygeos.Geometry(wkt))
    geom = pygeos.from_wkb(wkb)
    assert pygeos.is_geometry(geom).all()
    assert pygeos.is_empty(geom).all()
    assert pygeos.to_wkb(geom) == wkb


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


def test_to_wkt_point_empty():
    assert pygeos.to_wkt(empty_point) == "POINT EMPTY"


def test_to_wkt_geometrycollection_with_point_empty():
    collection = pygeos.geometrycollections([empty_point, point])
    # do not check the full value as some GEOS versions give
    # GEOMETRYCOLLECTION Z (...) and others give GEOMETRYCOLLECTION (...)
    assert pygeos.to_wkt(collection).endswith("(POINT EMPTY, POINT (2 3))")


def test_to_wkt_multipoint_with_point_empty_errors():
    # Test if segfault is prevented
    geom = pygeos.multipoints([empty_point, point])
    with pytest.raises(ValueError):
        pygeos.to_wkt(geom)


def test_repr():
    assert repr(point) == "<pygeos.Geometry POINT (2 3)>"


def test_repr_max_length():
    # the repr is limited to 80 characters
    geom = pygeos.linestrings(np.arange(1000), np.arange(1000))
    representation = repr(geom)
    assert len(representation) == 80
    assert representation.endswith("...>")


def test_repr_multipoint_with_point_empty():
    # Test if segfault is prevented
    geom = pygeos.multipoints([point, empty_point])
    assert repr(geom) == "<pygeos.Geometry Exception in WKT writer>"


def test_to_wkb():
    point = pygeos.points(1, 1)
    actual = pygeos.to_wkb(point, byte_order=1)
    assert actual == POINT11_WKB


def test_to_wkb_hex():
    point = pygeos.points(1, 1)
    actual = pygeos.to_wkb(point, hex=True, byte_order=1)
    le = "01"
    point_type = "01000000"
    coord = "000000000000F03F"  # 1.0 as double (LE)
    assert actual == le + point_type + 2 * coord


def test_to_wkb_3D():
    point_z = pygeos.points(1, 1, 1)
    actual = pygeos.to_wkb(point_z, byte_order=1)
    # fmt: off
    assert actual == b"\x01\x01\x00\x00\x80\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?"  # noqa
    # fmt: on
    actual = pygeos.to_wkb(point_z, output_dimension=2, byte_order=1)
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

    assert pygeos.to_wkb(actual, hex=True, byte_order=1) == wkb
    assert pygeos.to_wkb(actual, hex=True, include_srid=True, byte_order=1) == ewkb

    point = pygeos.points(1, 1)
    point_with_srid = pygeos.set_srid(point, np.int32(4326))
    result = pygeos.to_wkb(point_with_srid, include_srid=True, byte_order=1)
    assert np.frombuffer(result[5:9], "<u4").item() == 4326


@pytest.mark.skipif(
    pygeos.geos_version >= (3, 8, 0), reason="Pre GEOS 3.8.0 has 3D empty points"
)
@pytest.mark.parametrize(
    "geom,dims,expected",
    [
        (empty_point, 2, POINT_NAN_WKB),
        (empty_point, 3, POINTZ_NAN_WKB),
        (pygeos.multipoints([empty_point]), 2, MULTIPOINT_NAN_WKB),
        (pygeos.multipoints([empty_point]), 3, MULTIPOINTZ_NAN_WKB),
        (pygeos.geometrycollections([empty_point]), 2, GEOMETRYCOLLECTION_NAN_WKB),
        (pygeos.geometrycollections([empty_point]), 3, GEOMETRYCOLLECTIONZ_NAN_WKB),
        (
            pygeos.geometrycollections([pygeos.multipoints([empty_point])]),
            2,
            NESTED_COLLECTION_NAN_WKB,
        ),
        (
            pygeos.geometrycollections([pygeos.multipoints([empty_point])]),
            3,
            NESTED_COLLECTIONZ_NAN_WKB,
        ),
    ],
)
def test_to_wkb_point_empty_pre_geos38(geom, dims, expected):
    actual = pygeos.to_wkb(geom, output_dimension=dims, byte_order=1)
    # Use numpy.isnan; there are many byte representations for NaN
    assert actual[: -dims * 8] == expected[: -dims * 8]
    assert np.isnan(struct.unpack("<{}d".format(dims), actual[-dims * 8 :])).all()


@pytest.mark.skipif(
    pygeos.geos_version < (3, 8, 0), reason="Post GEOS 3.8.0 has 2D empty points"
)
@pytest.mark.parametrize(
    "geom,dims,expected",
    [
        (empty_point, 2, POINT_NAN_WKB),
        (empty_point, 3, POINT_NAN_WKB),
        (pygeos.multipoints([empty_point]), 2, MULTIPOINT_NAN_WKB),
        (pygeos.multipoints([empty_point]), 3, MULTIPOINT_NAN_WKB),
        (pygeos.geometrycollections([empty_point]), 2, GEOMETRYCOLLECTION_NAN_WKB),
        (pygeos.geometrycollections([empty_point]), 3, GEOMETRYCOLLECTION_NAN_WKB),
        (
            pygeos.geometrycollections([pygeos.multipoints([empty_point])]),
            2,
            NESTED_COLLECTION_NAN_WKB,
        ),
        (
            pygeos.geometrycollections([pygeos.multipoints([empty_point])]),
            3,
            NESTED_COLLECTION_NAN_WKB,
        ),
    ],
)
def test_to_wkb_point_empty_post_geos38(geom, dims, expected):
    # Post GEOS 3.8: empty point is 2D
    actual = pygeos.to_wkb(geom, output_dimension=dims, byte_order=1)
    # Use numpy.isnan; there are many byte representations for NaN
    assert actual[: -2 * 8] == expected[: -2 * 8]
    assert np.isnan(struct.unpack("<2d", actual[-2 * 8 :])).all()


@pytest.mark.parametrize(
    "wkb,expected_type",
    [
        (POINT_NAN_WKB, 0),
        (POINTZ_NAN_WKB, 0),
        (MULTIPOINT_NAN_WKB, 4),
        (MULTIPOINTZ_NAN_WKB, 4),
        (GEOMETRYCOLLECTION_NAN_WKB, 7),
        (GEOMETRYCOLLECTIONZ_NAN_WKB, 7),
        (NESTED_COLLECTION_NAN_WKB, 7),
        (NESTED_COLLECTIONZ_NAN_WKB, 7),
    ],
)
def test_from_wkb_point_empty(wkb, expected_type):
    geom = pygeos.from_wkb(wkb)
    # POINT (nan nan) transforms to an empty point
    # Note that the dimensionality (2D/3D) is GEOS-version dependent
    assert pygeos.is_empty(geom)
    assert pygeos.get_type_id(geom) == expected_type


def test_to_wkb_point_empty_srid():
    expected = pygeos.set_srid(empty_point, 4236)
    wkb = pygeos.to_wkb(expected, include_srid=True)
    actual = pygeos.from_wkb(wkb)
    assert pygeos.get_srid(actual) == 4236


@pytest.mark.parametrize("geom", all_types)
@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.ShapelyPreparedGeometry", ShapelyPreparedMock)
@mock.patch("pygeos.io.shapely_compatible", True)
@mock.patch("pygeos.io._shapely_checked", True)
def test_from_shapely(geom):
    actual = pygeos.from_shapely(ShapelyGeometryMock(geom))
    assert isinstance(actual, pygeos.Geometry)
    assert pygeos.equals(geom, actual)
    assert geom._ptr != actual._ptr


@pytest.mark.parametrize("geom", all_types)
@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.ShapelyPreparedGeometry", ShapelyPreparedMock)
@mock.patch("pygeos.io.shapely_compatible", True)
@mock.patch("pygeos.io._shapely_checked", True)
def test_from_shapely_prepared(geom):
    actual = pygeos.from_shapely(ShapelyPreparedMock(geom))
    assert isinstance(actual, pygeos.Geometry)
    assert pygeos.equals(geom, actual)
    assert geom._ptr != actual._ptr


@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.ShapelyPreparedGeometry", ShapelyPreparedMock)
@mock.patch("pygeos.io.shapely_compatible", True)
@mock.patch("pygeos.io._shapely_checked", True)
def test_from_shapely_arr():
    actual = pygeos.from_shapely([ShapelyGeometryMock(point), None])
    assert pygeos.equals(point, actual[0])


@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.ShapelyPreparedGeometry", ShapelyPreparedMock)
@mock.patch("pygeos.io.shapely_compatible", True)
@mock.patch("pygeos.io._shapely_checked", True)
def test_from_shapely_none():
    actual = pygeos.from_shapely(None)
    assert actual is None


@pytest.mark.parametrize("geom", [1, 2.3, "x", ShapelyGeometryMock(None)])
@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.ShapelyPreparedGeometry", ShapelyPreparedMock)
@mock.patch("pygeos.io.shapely_compatible", True)
@mock.patch("pygeos.io._shapely_checked", True)
def test_from_shapely_error(geom):
    with pytest.raises(TypeError):
        pygeos.from_shapely(geom)


@pytest.mark.parametrize("geom", all_types)
@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.ShapelyPreparedGeometry", ShapelyPreparedMock)
@mock.patch("pygeos.io.shapely_compatible", False)
@mock.patch("pygeos.io._shapely_checked", True)
def test_from_shapely_incompatible(geom):
    actual = pygeos.from_shapely(ShapelyGeometryMock(geom))
    assert isinstance(actual, pygeos.Geometry)
    assert pygeos.equals(geom, actual)
    assert geom._ptr != actual._ptr


@pytest.mark.parametrize("geom", all_types)
@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.ShapelyPreparedGeometry", ShapelyPreparedMock)
@mock.patch("pygeos.io.shapely_compatible", False)
@mock.patch("pygeos.io._shapely_checked", True)
def test_from_shapely_incompatible_prepared(geom):
    actual = pygeos.from_shapely(ShapelyPreparedMock(geom))
    assert isinstance(actual, pygeos.Geometry)
    assert pygeos.equals(geom, actual)
    assert geom._ptr != actual._ptr


@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.ShapelyPreparedGeometry", ShapelyPreparedMock)
@mock.patch("pygeos.io.shapely_compatible", False)
@mock.patch("pygeos.io._shapely_checked", True)
def test_from_shapely_incompatible_none():
    actual = pygeos.from_shapely(None)
    assert actual is None


@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.ShapelyPreparedGeometry", ShapelyPreparedMock)
@mock.patch("pygeos.io.shapely_compatible", False)
@mock.patch("pygeos.io._shapely_checked", True)
def test_from_shapely_incompatible_array():
    actual = pygeos.from_shapely([ShapelyGeometryMock(point), None])
    assert pygeos.equals(point, actual[0])


@pytest.mark.parametrize("geom", all_types)
@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.shapely_wkb_loads", shapely_wkb_loads_mock)
@mock.patch("pygeos.io.shapely_compatible", False)
@mock.patch("pygeos.io._shapely_checked", True)
def test_to_shapely_incompatible(geom):
    actual = pygeos.to_shapely(geom)
    assert isinstance(actual, ShapelyGeometryMock)
    assert pygeos.equals(geom, actual.g)
    assert geom._ptr != actual.g._ptr


@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.shapely_wkb_loads", shapely_wkb_loads_mock)
@mock.patch("pygeos.io.shapely_compatible", False)
@mock.patch("pygeos.io._shapely_checked", True)
def test_to_shapely_incompatible_none():
    actual = pygeos.to_shapely(None)
    assert actual is None


@mock.patch("pygeos.io.ShapelyGeometry", ShapelyGeometryMock)
@mock.patch("pygeos.io.shapely_wkb_loads", shapely_wkb_loads_mock)
@mock.patch("pygeos.io.shapely_compatible", False)
@mock.patch("pygeos.io._shapely_checked", True)
def test_to_shapely_incompatible_array():
    actual = pygeos.to_shapely([point, None])
    assert pygeos.equals(point, actual[0].g)


@pytest.mark.parametrize("geom", all_types + (point_z, empty_point))
def test_pickle(geom):
    if pygeos.get_type_id(geom) == 2:
        # Linearrings get converted to linestrings
        expected = pygeos.linestrings(pygeos.get_coordinates(geom))
    else:
        expected = geom
    pickled = pickle.dumps(geom)
    assert pygeos.equals_exact(pickle.loads(pickled), expected)


def test_pickle_with_srid():
    geom = pygeos.set_srid(point, 4326)
    pickled = pickle.dumps(geom)
    assert pygeos.get_srid(pickle.loads(pickled)) == 4326
