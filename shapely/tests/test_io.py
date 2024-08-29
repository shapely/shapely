import json
import pickle
import struct
import warnings

import numpy as np
import pytest

import shapely
from shapely import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
    all_types,
    all_types_m,
    all_types_z,
    all_types_zm,
    empty_point,
    empty_point_m,
    empty_point_z,
    empty_point_zm,
    equal_geometries_abnormally_yield_unequal,
    multi_point_empty,
    multi_point_empty_m,
    multi_point_empty_z,
    multi_point_empty_zm,
    point,
    point_m,
    point_z,
    point_zm,
    polygon_z,
)

EWKBZ = 0x80000000
EWKBM = 0x40000000
EWKBZM = EWKBZ | EWKBM
ISOWKBZ = 1000
ISOWKBM = 2000
ISOWKBZM = ISOWKBZ + ISOWKBM
POINT11_WKB = struct.pack("<BI2d", 1, 1, 1.0, 1.0)
NAN = struct.pack("<d", float("nan"))
POINT_NAN_WKB = struct.pack("<BI", 1, 1) + (NAN * 2)
POINTZ_NAN_WKB = struct.pack("<BI", 1, 1 | EWKBZ) + (NAN * 3)
POINTM_NAN_WKB = struct.pack("<BI", 1, 1 | EWKBM) + (NAN * 3)
POINTZM_NAN_WKB = struct.pack("<BI", 1, 1 | EWKBZM) + (NAN * 4)
MULTIPOINT_NAN_WKB = struct.pack("<BII", 1, 4, 1) + POINT_NAN_WKB
MULTIPOINTZ_NAN_WKB = struct.pack("<BII", 1, 4 | EWKBZ, 1) + POINTZ_NAN_WKB
MULTIPOINTM_NAN_WKB = struct.pack("<BII", 1, 4 | EWKBM, 1) + POINTM_NAN_WKB
MULTIPOINTZM_NAN_WKB = struct.pack("<BII", 1, 4 | EWKBZM, 1) + POINTZM_NAN_WKB
GEOMETRYCOLLECTION_NAN_WKB = struct.pack("<BII", 1, 7, 1) + POINT_NAN_WKB
GEOMETRYCOLLECTIONZ_NAN_WKB = struct.pack("<BII", 1, 7 | EWKBZ, 1) + POINTZ_NAN_WKB
GEOMETRYCOLLECTIONM_NAN_WKB = struct.pack("<BII", 1, 7 | EWKBM, 1) + POINTM_NAN_WKB
GEOMETRYCOLLECTIONZM_NAN_WKB = struct.pack("<BII", 1, 7 | EWKBZM, 1) + POINTZM_NAN_WKB
NESTED_COLLECTION_NAN_WKB = struct.pack("<BII", 1, 7, 1) + MULTIPOINT_NAN_WKB
NESTED_COLLECTIONZ_NAN_WKB = struct.pack("<BII", 1, 7 | EWKBZ, 1) + MULTIPOINTZ_NAN_WKB
NESTED_COLLECTIONM_NAN_WKB = struct.pack("<BII", 1, 7 | EWKBM, 1) + MULTIPOINTM_NAN_WKB
NESTED_COLLECTIONZM_NAN_WKB = (
    struct.pack("<BII", 1, 7 | EWKBZM, 1) + MULTIPOINTZM_NAN_WKB
)
INVALID_WKB = "01030000000100000002000000507daec600b1354100de02498e5e3d41306ea321fcb03541a011a53d905e3d41"  # noqa: E501

GEOJSON_GEOMETRY = json.dumps({"type": "Point", "coordinates": [125.6, 10.1]}, indent=4)
GEOJSON_FEATURE = json.dumps(
    {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [125.6, 10.1]},
        "properties": {"name": "Dinagat Islands"},
    },
    indent=4,
)
GEOJSON_FEATURECOLECTION = json.dumps(
    {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [102.0, 0.6]},
                "properties": {"prop0": "value0"},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [102.0, 0.0],
                        [103.0, 1.0],
                        [104.0, 0.0],
                        [105.0, 1.0],
                    ],
                },
                "properties": {"prop1": 0.0, "prop0": "value0"},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [100.0, 0.0],
                            [101.0, 0.0],
                            [101.0, 1.0],
                            [100.0, 1.0],
                            [100.0, 0.0],
                        ]
                    ],
                },
                "properties": {"prop1": {"this": "that"}, "prop0": "value0"},
            },
        ],
    },
    indent=4,
)

GEOJSON_GEOMETRY_EXPECTED = shapely.points(125.6, 10.1)
GEOJSON_COLLECTION_EXPECTED = [
    shapely.points([102.0, 0.6]),
    shapely.linestrings([[102.0, 0.0], [103.0, 1.0], [104.0, 0.0], [105.0, 1.0]]),
    shapely.polygons(
        [[100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0]]
    ),
]


def test_from_wkt():
    expected = shapely.points(1, 1)
    actual = shapely.from_wkt("POINT (1 1)")
    assert_geometries_equal(actual, expected)
    # also accept bytes
    actual = shapely.from_wkt(b"POINT (1 1)")
    assert_geometries_equal(actual, expected)


def test_from_wkt_none():
    # None propagates
    assert shapely.from_wkt(None) is None


def test_from_wkt_exceptions():
    with pytest.raises(TypeError, match="Expected bytes or string, got int"):
        shapely.from_wkt(1)

    with pytest.raises(
        shapely.GEOSException, match="Expected word but encountered end of stream"
    ):
        shapely.from_wkt("")

    with pytest.raises(shapely.GEOSException, match="Unknown type: 'NOT'"):
        shapely.from_wkt("NOT A WKT STRING")


def test_from_wkt_warn_on_invalid():
    with pytest.warns(Warning, match="Invalid WKT"):
        shapely.from_wkt("", on_invalid="warn")

    with pytest.warns(Warning, match="Invalid WKT"):
        shapely.from_wkt("NOT A WKT STRING", on_invalid="warn")


def test_from_wkb_ignore_on_invalid():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shapely.from_wkt("", on_invalid="ignore")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shapely.from_wkt("NOT A WKT STRING", on_invalid="ignore")


def test_from_wkt_on_invalid_unsupported_option():
    with pytest.raises(ValueError, match="not a valid option"):
        shapely.from_wkt("NOT A WKT STRING", on_invalid="unsupported_option")


@pytest.mark.parametrize("geom", all_types)
def test_from_wkt_all_types(geom):
    if geom.is_empty and shapely.get_num_geometries(geom) > 0:
        # Older GEOS versions have various issues
        if shapely.geos_version < (3, 9, 0) and geom.geom_type == "MultiPoint":
            with pytest.raises(ValueError):
                shapely.to_wkt(geom)
            pytest.skip(
                "GEOS < 3.9.0 does not support WKT of multipoint with empty points"
            )

    wkt = shapely.to_wkt(geom)
    actual = shapely.from_wkt(wkt)

    if equal_geometries_abnormally_yield_unequal(geom):
        # check abnormal test
        with pytest.raises(AssertionError):
            assert_geometries_equal(actual, geom)
    else:
        # normal test
        assert_geometries_equal(actual, geom)


@pytest.mark.parametrize(
    "wkt",
    ("POINT EMPTY", "LINESTRING EMPTY", "POLYGON EMPTY", "GEOMETRYCOLLECTION EMPTY"),
)
def test_from_wkt_empty(wkt):
    geom = shapely.from_wkt(wkt)
    assert shapely.is_geometry(geom).all()
    assert shapely.is_empty(geom).all()
    assert shapely.to_wkt(geom) == wkt


def test_from_wkb():
    expected = shapely.points(1, 1)
    actual = shapely.from_wkb(POINT11_WKB)
    assert_geometries_equal(actual, expected)


def test_from_wkb_hex():
    # HEX form
    expected = shapely.points(1, 1)
    actual = shapely.from_wkb("0101000000000000000000F03F000000000000F03F")
    assert_geometries_equal(actual, expected)
    actual = shapely.from_wkb(b"0101000000000000000000F03F000000000000F03F")
    assert_geometries_equal(actual, expected)


def test_from_wkb_none():
    # None propagates
    assert shapely.from_wkb(None) is None


def test_from_wkb_exceptions():
    with pytest.raises(TypeError, match="Expected bytes or string, got int"):
        shapely.from_wkb(1)

    # invalid WKB
    with pytest.raises(
        shapely.GEOSException,
        match=(
            "Unexpected EOF parsing WKB|"
            "ParseException: Input buffer is smaller than requested object size"
        ),
    ):
        result = shapely.from_wkb(b"\x01\x01\x00\x00\x00\x00")
        assert result is None

    # invalid ring in WKB
    with pytest.raises(
        shapely.GEOSException,
        match="Points of LinearRing do not form a closed linestring",
    ):
        result = shapely.from_wkb(INVALID_WKB)
        assert result is None


def test_from_wkb_warn_on_invalid_warn():
    # invalid WKB
    with pytest.warns(Warning, match="Invalid WKB"):
        result = shapely.from_wkb(b"\x01\x01\x00\x00\x00\x00", on_invalid="warn")
        assert result is None

    # invalid ring in WKB
    with pytest.warns(Warning, match="Invalid WKB"):
        result = shapely.from_wkb(INVALID_WKB, on_invalid="warn")
        assert result is None


def test_from_wkb_ignore_on_invalid_ignore():
    # invalid WKB
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning
        result = shapely.from_wkb(b"\x01\x01\x00\x00\x00\x00", on_invalid="ignore")
        assert result is None

    # invalid ring in WKB
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning
        result = shapely.from_wkb(INVALID_WKB, on_invalid="ignore")
        assert result is None


def test_from_wkb_on_invalid_unsupported_option():
    with pytest.raises(ValueError, match="not a valid option"):
        shapely.from_wkb(b"\x01\x01\x00\x00\x00\x00", on_invalid="unsupported_option")


@pytest.mark.parametrize("geom", all_types)
@pytest.mark.parametrize("use_hex", [False, True])
@pytest.mark.parametrize("byte_order", [0, 1])
def test_from_wkb_all_types(geom, use_hex, byte_order):
    if shapely.get_type_id(geom) == shapely.GeometryType.LINEARRING:
        pytest.skip("Linearrings are not preserved in WKB")
    wkb = shapely.to_wkb(geom, hex=use_hex, byte_order=byte_order)
    actual = shapely.from_wkb(wkb)
    assert_geometries_equal(actual, geom)


@pytest.mark.parametrize("geom", all_types_z)
@pytest.mark.parametrize("use_hex", [False, True])
@pytest.mark.parametrize("byte_order", [0, 1])
def test_from_wkb_all_types_z(geom, use_hex, byte_order):
    if shapely.get_type_id(geom) == shapely.GeometryType.LINEARRING:
        pytest.skip("Linearrings are not preserved in WKB")
    wkb = shapely.to_wkb(geom, hex=use_hex, byte_order=byte_order)
    actual = shapely.from_wkb(wkb)
    assert_geometries_equal(actual, geom)


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="M coordinates not supported with GEOS < 3.12",
)
@pytest.mark.parametrize("geom", all_types_m)
@pytest.mark.parametrize("use_hex", [False, True])
@pytest.mark.parametrize("byte_order", [0, 1])
def test_from_wkb_all_types_m(geom, use_hex, byte_order):
    if shapely.get_type_id(geom) == shapely.GeometryType.LINEARRING:
        pytest.skip("Linearrings are not preserved in WKB")
    wkb = shapely.to_wkb(geom, hex=use_hex, byte_order=byte_order)
    actual = shapely.from_wkb(wkb)
    assert_geometries_equal(actual, geom)


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="M coordinates not supported with GEOS < 3.12",
)
@pytest.mark.parametrize("geom", all_types_zm)
@pytest.mark.parametrize("use_hex", [False, True])
@pytest.mark.parametrize("byte_order", [0, 1])
def test_from_wkb_all_types_zm(geom, use_hex, byte_order):
    if shapely.get_type_id(geom) == shapely.GeometryType.LINEARRING:
        pytest.skip("Linearrings are not preserved in WKB")
    wkb = shapely.to_wkb(geom, hex=use_hex, byte_order=byte_order)
    actual = shapely.from_wkb(wkb)
    assert_geometries_equal(actual, geom)


@pytest.mark.parametrize(
    "geom",
    (Point(), LineString(), Polygon(), GeometryCollection()),
)
def test_from_wkb_empty(geom):
    wkb = shapely.to_wkb(geom)
    geom = shapely.from_wkb(wkb)
    assert shapely.is_geometry(geom).all()
    assert shapely.is_empty(geom).all()
    assert shapely.to_wkb(geom) == wkb


def test_to_wkt():
    point = shapely.points(1, 1)
    actual = shapely.to_wkt(point)
    assert actual == "POINT (1 1)"

    actual = shapely.to_wkt(point, trim=False)
    assert actual == "POINT (1.000000 1.000000)"

    actual = shapely.to_wkt(point, rounding_precision=3, trim=False)
    assert actual == "POINT (1.000 1.000)"


def test_to_wkt_z():
    point = shapely.points(1, 2, 3)

    assert shapely.to_wkt(point) == "POINT Z (1 2 3)"
    assert shapely.to_wkt(point, output_dimension=2) == "POINT (1 2)"
    assert shapely.to_wkt(point, output_dimension=3) == "POINT Z (1 2 3)"
    assert shapely.to_wkt(point, old_3d=True) == "POINT (1 2 3)"

    if shapely.geos_version >= (3, 12, 0):
        assert shapely.to_wkt(point, output_dimension=4) == "POINT Z (1 2 3)"


def test_to_wkt_m():
    point = shapely.from_wkt("POINT M (1 2 4)")

    assert shapely.to_wkt(point, output_dimension=2) == "POINT (1 2)"

    if shapely.geos_version < (3, 12, 0):
        # previous behavior was to incorrectly parse M as Z
        assert shapely.to_wkt(point) == "POINT Z (1 2 4)"
        assert shapely.to_wkt(point, output_dimension=3) == "POINT Z (1 2 4)"
        assert shapely.to_wkt(point, old_3d=True) == "POINT (1 2 4)"
    else:
        assert shapely.to_wkt(point) == "POINT M (1 2 4)"
        assert shapely.to_wkt(point, output_dimension=3) == "POINT M (1 2 4)"
        assert shapely.to_wkt(point, output_dimension=4) == "POINT M (1 2 4)"
        assert shapely.to_wkt(point, old_3d=True) == "POINT M (1 2 4)"


def test_to_wkt_zm():
    point = shapely.from_wkt("POINT ZM (1 2 3 4)")

    assert shapely.to_wkt(point, output_dimension=2) == "POINT (1 2)"
    assert shapely.to_wkt(point, output_dimension=3) == "POINT Z (1 2 3)"

    if shapely.geos_version < (3, 12, 0):
        # previous behavior was to parse and ignore M
        assert shapely.to_wkt(point) == "POINT Z (1 2 3)"
        assert shapely.to_wkt(point, old_3d=True) == "POINT (1 2 3)"
    else:
        assert shapely.to_wkt(point) == "POINT ZM (1 2 3 4)"
        assert shapely.to_wkt(point, output_dimension=4) == "POINT ZM (1 2 3 4)"
        assert shapely.to_wkt(point, old_3d=True) == "POINT (1 2 3 4)"


def test_to_wkt_none():
    # None propagates
    assert shapely.to_wkt(None) is None


def test_to_wkt_array_with_empty_z():
    # See GH-2004
    empty_wkt = ["POINT Z EMPTY", None, "POLYGON Z EMPTY"]
    empty_geoms = shapely.from_wkt(empty_wkt)
    if shapely.geos_version < (3, 9, 0):
        empty_wkt = ["POINT EMPTY", None, "POLYGON EMPTY"]
    assert list(shapely.to_wkt(empty_geoms)) == empty_wkt


def test_to_wkt_exceptions():
    with pytest.raises(TypeError):
        shapely.to_wkt(1)

    with pytest.raises(shapely.GEOSException):
        shapely.to_wkt(point, output_dimension=5)


def test_to_wkt_point_empty():
    assert shapely.to_wkt(empty_point) == "POINT EMPTY"


@pytest.mark.skipif(
    shapely.geos_version < (3, 9, 0),
    reason="Empty geometries have no dimensionality on GEOS < 3.9",
)
@pytest.mark.parametrize(
    "wkt",
    [
        "POINT Z EMPTY",
        "LINESTRING Z EMPTY",
        "LINEARRING Z EMPTY",
        "POLYGON Z EMPTY",
    ],
)
def test_to_wkt_empty_z(wkt):
    assert shapely.to_wkt(shapely.from_wkt(wkt)) == wkt


def test_to_wkt_geometrycollection_with_point_empty():
    collection = shapely.geometrycollections([empty_point, point])
    # do not check the full value as some GEOS versions give
    # GEOMETRYCOLLECTION Z (...) and others give GEOMETRYCOLLECTION (...)
    assert shapely.to_wkt(collection).endswith("(POINT EMPTY, POINT (2 3))")


@pytest.mark.skipif(
    shapely.geos_version < (3, 9, 0),
    reason="MULTIPOINT (EMPTY, (2 3)) only works for GEOS >= 3.9",
)
def test_to_wkt_multipoint_with_point_empty():
    geom = shapely.multipoints([empty_point, point])
    if shapely.geos_version >= (3, 12, 0):
        expected = "MULTIPOINT (EMPTY, (2 3))"
    else:
        # invalid WKT form
        expected = "MULTIPOINT (EMPTY, 2 3)"
    assert shapely.to_wkt(geom) == expected


@pytest.mark.skipif(
    shapely.geos_version >= (3, 9, 0),
    reason="MULTIPOINT (EMPTY, 2 3) gives ValueError on GEOS < 3.9",
)
def test_to_wkt_multipoint_with_point_empty_errors():
    # test if segfault is prevented
    geom = shapely.multipoints([empty_point, point])
    with pytest.raises(ValueError):
        shapely.to_wkt(geom)


@pytest.mark.parametrize("geom", [Point(1e100, 0), Point(0, 1e100)])
def test_to_wkt_large_float_ok(geom):
    # https://github.com/shapely/shapely/issues/1903
    shapely.to_wkt(geom)
    assert "Exception in WKT writer" not in repr(geom)


@pytest.mark.parametrize("geom", [Point(1e101, 0), Point(0, 1e101)])
def test_to_wkt_large_float(geom):
    if shapely.geos_version >= (3, 13, 0):
        # round-trip WKT
        assert geom.equals(shapely.from_wkt(shapely.to_wkt(geom)))
    else:
        # https://github.com/shapely/shapely/issues/1903
        with pytest.raises(
            ValueError, match="WKT output of coordinates greater than.*"
        ):
            shapely.to_wkt(geom)
        assert "Exception in WKT writer" in repr(geom)


@pytest.mark.parametrize(
    "geom",
    [
        # We implemented our own "GetZMax", so go through all geometry types:
        Point(0, 0, 1e101),
        LineString([(0, 0, 0), (0, 0, 1e101)]),
        LinearRing([(0, 0, 0), (0, 1, 0), (1, 0, 1e101), (0, 0, 0)]),
        Polygon([(0, 0, 0), (0, 1, 0), (1, 0, 1e101), (0, 0, 0)]),
        Polygon(
            [(0, 0, 0), (0, 10, 0), (10, 0, 0), (0, 0, 0)],
            [[(0, 0, 0), (0, 1, 0), (1, 0, 1e101), (0, 0, 0)]],
        ),
        MultiPoint([(0, 0, 0), (0, 0, 1e101)]),
        MultiLineString(
            [LineString([(0, 0, 0), (0, 1, 0)]), LineString([(0, 0, 0), (0, 1, 1e101)])]
        ),
        MultiPolygon(
            [polygon_z, Polygon([(0, 0, 0), (0, 1, 0), (1, 0, 1e101), (0, 0, 0)])]
        ),
        GeometryCollection([point_z, Point(0, 0, 1e101)]),
        GeometryCollection([GeometryCollection([Point(0, 0, 1e101)])]),
        LineString([(0, 0, np.nan), (0, 0, 1e101)]),
        Polygon([(0, 0, np.nan), (0, 1, 0), (1, 0, 1e101), (0, 0, 0)]),
        GeometryCollection([Point(0, 0), Point(0, 0, 1e101)]),
    ],
)
def test_to_wkt_large_float_3d_no_crash(geom):
    # https://github.com/shapely/shapely/issues/1903
    # just test if there is a crash (detailed behaviour differs per GEOS version)
    try:
        shapely.to_wkt(geom)
    except ValueError as e:
        assert str(e).startswith("WKT output of coordinates greater than")
    repr(geom)


def test_to_wkt_large_float_skip_z():
    # https://github.com/shapely/shapely/issues/1903
    assert shapely.to_wkt(Point(0, 0, 1e101), output_dimension=2) == "POINT (0 0)"


def test_to_wkt_large_float_no_trim():
    # https://github.com/shapely/shapely/issues/1903
    # don't test the exact number, it is ridiculously large and probably platform
    # dependent
    assert shapely.to_wkt(Point(1e101, 0), trim=False).startswith("POINT (")


def test_repr():
    assert repr(point) == "<POINT (2 3)>"
    assert repr(point_z) == "<POINT Z (2 3 4)>"


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="M coordinates not supported with GEOS < 3.12",
)
def test_repr_m():
    assert repr(point_m) == "<POINT M (2 3 5)>"
    assert repr(point_zm) == "<POINT ZM (2 3 4 5)>"


def test_repr_max_length():
    # the repr is limited to 80 characters
    geom = shapely.linestrings(np.arange(1000), np.arange(1000))
    representation = repr(geom)
    assert len(representation) == 80
    assert representation.endswith("...>")


@pytest.mark.skipif(
    shapely.geos_version >= (3, 9, 0),
    reason="MULTIPOINT (EMPTY, 2 3) gives Exception on GEOS < 3.9",
)
def test_repr_multipoint_with_point_empty():
    # Test if segfault is prevented
    geom = shapely.multipoints([point, empty_point])
    assert repr(geom) == "<shapely.MultiPoint Exception in WKT writer>"


@pytest.mark.skipif(
    shapely.geos_version < (3, 9, 0),
    reason="Empty geometries have no dimensionality on GEOS < 3.9",
)
def test_repr_point_z_empty():
    assert repr(empty_point_z) == "<POINT Z EMPTY>"


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="M coordinates not supported with GEOS < 3.12",
)
def test_repr_point_m_empty():
    assert repr(empty_point_m) == "<POINT M EMPTY>"
    assert repr(empty_point_zm) == "<POINT ZM EMPTY>"


def test_to_wkb():
    point = shapely.points(1, 1)
    actual = shapely.to_wkb(point, byte_order=1)
    assert actual == POINT11_WKB


def test_to_wkb_hex():
    point = shapely.points(1, 1)
    actual = shapely.to_wkb(point, hex=True, byte_order=1)
    le = "01"
    point_type = "01000000"
    coord = "000000000000F03F"  # 1.0 as double (LE)
    assert actual == le + point_type + 2 * coord


def test_to_wkb_z():
    point = shapely.points(1, 2, 3)

    expected_wkb = struct.pack("<BI2d", 1, 1, 1.0, 2.0)
    expected_wkb_z = struct.pack("<BI3d", 1, 1 | EWKBZ, 1.0, 2.0, 3.0)

    assert shapely.to_wkb(point, byte_order=1) == expected_wkb_z
    assert shapely.to_wkb(point, output_dimension=2, byte_order=1) == expected_wkb
    assert shapely.to_wkb(point, output_dimension=3, byte_order=1) == expected_wkb_z
    if shapely.geos_version >= (3, 12, 0):
        assert shapely.to_wkb(point, output_dimension=4, byte_order=1) == expected_wkb_z


def test_to_wkb_m():
    # POINT M (1 2 4)
    point = shapely.from_wkb(struct.pack("<BI3d", 1, 1 | EWKBM, 1.0, 2.0, 4.0))

    expected_wkb = struct.pack("<BI2d", 1, 1, 1.0, 2.0)
    expected_wkb_m = struct.pack("<BI3d", 1, 1 | EWKBM, 1.0, 2.0, 4.0)
    if shapely.geos_version < (3, 12, 0):
        # previous behavior was to ignore M, treat as 2D
        expected_wkb_m = expected_wkb

    assert shapely.to_wkb(point, byte_order=1) == expected_wkb_m
    assert shapely.to_wkb(point, output_dimension=2, byte_order=1) == expected_wkb
    assert shapely.to_wkb(point, output_dimension=3, byte_order=1) == expected_wkb_m
    if shapely.geos_version >= (3, 12, 0):
        assert shapely.to_wkb(point, output_dimension=4, byte_order=1) == expected_wkb_m


def test_to_wkb_zm():
    # POINT ZM (1 2 3 4)
    point = shapely.from_wkb(struct.pack("<BI4d", 1, 1 | EWKBZM, 1.0, 2.0, 3.0, 4.0))

    expected_wkb = struct.pack("<BI2d", 1, 1, 1.0, 2.0)
    expected_wkb_z = struct.pack("<BI3d", 1, 1 | EWKBZ, 1.0, 2.0, 3.0)
    expected_wkb_zm = struct.pack("<BI4d", 1, 1 | EWKBZM, 1.0, 2.0, 3.0, 4.0)
    if shapely.geos_version < (3, 12, 0):
        # previous behavior was to ignore M, treat as XYZ
        expected_wkb_zm = expected_wkb_z

    assert shapely.to_wkb(point, byte_order=1) == expected_wkb_zm
    assert shapely.to_wkb(point, output_dimension=2, byte_order=1) == expected_wkb
    assert shapely.to_wkb(point, output_dimension=3, byte_order=1) == expected_wkb_z
    if shapely.geos_version >= (3, 12, 0):
        assert (
            shapely.to_wkb(point, output_dimension=4, byte_order=1) == expected_wkb_zm
        )


def test_to_wkb_none():
    # None propagates
    assert shapely.to_wkb(None) is None


def test_to_wkb_exceptions():
    with pytest.raises(TypeError):
        shapely.to_wkb(1)

    with pytest.raises(shapely.GEOSException):
        shapely.to_wkb(point, output_dimension=5)

    with pytest.raises(ValueError):
        shapely.to_wkb(point, flavor="other")


def test_to_wkb_byte_order():
    point = shapely.points(1.0, 1.0)
    be = b"\x00"
    le = b"\x01"
    point_type = b"\x01\x00\x00\x00"  # 1 as 32-bit uint (LE)
    coord = b"\x00\x00\x00\x00\x00\x00\xf0?"  # 1.0 as double (LE)

    assert shapely.to_wkb(point, byte_order=1) == le + point_type + 2 * coord
    assert (
        shapely.to_wkb(point, byte_order=0) == be + point_type[::-1] + 2 * coord[::-1]
    )


def test_to_wkb_srid():
    # hex representation of POINT (0 0) with SRID=4
    ewkb = "01010000200400000000000000000000000000000000000000"
    wkb = "010100000000000000000000000000000000000000"

    actual = shapely.from_wkb(ewkb)
    assert shapely.to_wkt(actual, trim=True) == "POINT (0 0)"

    assert shapely.to_wkb(actual, hex=True, byte_order=1) == wkb
    assert shapely.to_wkb(actual, hex=True, include_srid=True, byte_order=1) == ewkb

    point = shapely.points(1, 1)
    point_with_srid = shapely.set_srid(point, np.int32(4326))
    result = shapely.to_wkb(point_with_srid, include_srid=True, byte_order=1)
    assert np.frombuffer(result[5:9], "<u4").item() == 4326


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10.0")
def test_to_wkb_flavor():
    # http://libgeos.org/specifications/wkb/#extended-wkb
    actual = shapely.to_wkb(point_z, byte_order=1)  # default "extended"
    assert actual.hex()[2:10] == struct.pack("<I", 1 | EWKBZ).hex()
    actual = shapely.to_wkb(point_z, byte_order=1, flavor="extended")
    assert actual.hex()[2:10] == struct.pack("<I", 1 | EWKBZ).hex()
    actual = shapely.to_wkb(point_z, byte_order=1, flavor="iso")
    assert actual.hex()[2:10] == struct.pack("<I", 1 | ISOWKBZ).hex()


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="M coordinates not supported with GEOS < 3.12",
)
def test_to_wkb_m_flavor():
    # XYM
    actual = shapely.to_wkb(point_m, byte_order=1)  # default "extended"
    assert actual.hex()[2:10] == struct.pack("<I", 1 | EWKBM).hex()
    actual = shapely.to_wkb(point_m, byte_order=1, flavor="iso")
    assert actual.hex()[2:10] == struct.pack("<I", 1 | ISOWKBM).hex()

    # XYZM
    actual = shapely.to_wkb(point_zm, byte_order=1)  # default "extended"
    assert actual.hex()[2:10] == struct.pack("<I", 1 | EWKBZM).hex()
    actual = shapely.to_wkb(point_zm, byte_order=1, flavor="iso")
    assert actual.hex()[2:10] == struct.pack("<I", 1 | ISOWKBZM).hex()


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10.0")
def test_to_wkb_flavor_srid():
    with pytest.raises(ValueError, match="cannot be used together"):
        shapely.to_wkb(point_z, include_srid=True, flavor="iso")


@pytest.mark.skipif(shapely.geos_version >= (3, 10, 0), reason="GEOS < 3.10.0")
def test_to_wkb_flavor_unsupported_geos():
    with pytest.raises(UnsupportedGEOSVersionError):
        shapely.to_wkb(point_z, flavor="iso")


@pytest.mark.parametrize(
    "geom,expected",
    [
        pytest.param(empty_point, POINT_NAN_WKB, id="POINT EMPTY"),
        pytest.param(empty_point_z, POINT_NAN_WKB, id="POINT Z EMPTY"),
        pytest.param(empty_point_m, POINT_NAN_WKB, id="POINT M EMPTY"),
        pytest.param(empty_point_zm, POINT_NAN_WKB, id="POINT ZM EMPTY"),
        pytest.param(
            multi_point_empty,
            MULTIPOINT_NAN_WKB,
            id="MULTIPOINT EMPTY",
        ),
        pytest.param(
            multi_point_empty_z,
            MULTIPOINT_NAN_WKB,
            id="MULTIPOINT Z EMPTY",
        ),
        pytest.param(
            multi_point_empty_m,
            MULTIPOINT_NAN_WKB,
            id="MULTIPOINT M EMPTY",
        ),
        pytest.param(
            multi_point_empty_zm,
            MULTIPOINT_NAN_WKB,
            id="MULTIPOINT ZM EMPTY",
        ),
        pytest.param(
            shapely.geometrycollections([empty_point]),
            GEOMETRYCOLLECTION_NAN_WKB,
            id="GEOMETRYCOLLECTION (POINT EMPTY)",
        ),
        pytest.param(
            shapely.geometrycollections([empty_point_z]),
            GEOMETRYCOLLECTION_NAN_WKB,
            id="GEOMETRYCOLLECTION (POINT Z EMPTY)",
        ),
        pytest.param(
            shapely.geometrycollections([empty_point_m]),
            GEOMETRYCOLLECTION_NAN_WKB,
            id="GEOMETRYCOLLECTION (POINT M EMPTY)",
        ),
        pytest.param(
            shapely.geometrycollections([empty_point_zm]),
            GEOMETRYCOLLECTION_NAN_WKB,
            id="GEOMETRYCOLLECTION (POINT ZM EMPTY)",
        ),
        pytest.param(
            shapely.geometrycollections([multi_point_empty]),
            NESTED_COLLECTION_NAN_WKB,
            id="GEOMETRYCOLLECTION (MULTIPOINT EMPTY)",
        ),
        pytest.param(
            shapely.geometrycollections([multi_point_empty_z]),
            NESTED_COLLECTION_NAN_WKB,
            id="GEOMETRYCOLLECTION (MULTIPOINT Z EMPTY)",
        ),
        pytest.param(
            shapely.geometrycollections([multi_point_empty_m]),
            NESTED_COLLECTION_NAN_WKB,
            id="GEOMETRYCOLLECTION (MULTIPOINT M EMPTY)",
        ),
        pytest.param(
            shapely.geometrycollections([multi_point_empty_zm]),
            NESTED_COLLECTION_NAN_WKB,
            id="GEOMETRYCOLLECTION (MULTIPOINT ZM EMPTY)",
        ),
    ],
)
def test_to_wkb_point_empty_2d(geom, expected):
    actual = shapely.to_wkb(geom, output_dimension=2, byte_order=1)
    # Split 'actual' into header and coordinates
    coordinate_length = 16
    header_length = len(expected) - coordinate_length
    # Check the total length (this checks the correct dimensionality)
    assert len(actual) == header_length + coordinate_length
    # Check the header
    assert actual[:header_length] == expected[:header_length]
    # Check the coordinates (using numpy.isnan; there are many byte representations for
    # NaN)
    assert np.isnan(struct.unpack("<2d", actual[header_length:])).all()


@pytest.mark.xfail(
    shapely.geos_version[:2] == (3, 8), reason="GEOS==3.8 never outputs 3D empty points"
)
@pytest.mark.parametrize(
    "geom,expected",
    [
        pytest.param(empty_point_z, POINTZ_NAN_WKB, id="POINT Z EMPTY"),
        pytest.param(empty_point_zm, POINTZ_NAN_WKB, id="POINT ZM EMPTY"),
        pytest.param(
            multi_point_empty_z,
            MULTIPOINTZ_NAN_WKB,
            id="MULTIPOINT Z EMPTY",
        ),
        pytest.param(
            multi_point_empty_zm,
            MULTIPOINTZ_NAN_WKB,
            id="MULTIPOINT ZM EMPTY",
        ),
        pytest.param(
            shapely.geometrycollections([empty_point_z]),
            GEOMETRYCOLLECTIONZ_NAN_WKB,
            id="GEOMETRYCOLLECTION (POINT Z EMPTY)",
        ),
        pytest.param(
            shapely.geometrycollections([empty_point_zm]),
            GEOMETRYCOLLECTIONZ_NAN_WKB,
            id="GEOMETRYCOLLECTION (POINT ZM EMPTY)",
        ),
        pytest.param(
            shapely.geometrycollections([multi_point_empty_z]),
            NESTED_COLLECTIONZ_NAN_WKB,
            id="GEOMETRYCOLLECTION (MULTIPOINT Z EMPTY)",
        ),
        pytest.param(
            shapely.geometrycollections([multi_point_empty_zm]),
            NESTED_COLLECTIONZ_NAN_WKB,
            id="GEOMETRYCOLLECTION (MULTIPOINT ZM EMPTY)",
        ),
    ],
)
def test_to_wkb_point_empty_z(geom, expected):
    actual = shapely.to_wkb(geom, output_dimension=3, byte_order=1)
    # Split 'actual' into header and coordinates
    coordinate_length = 8 * 3
    header_length = len(expected) - coordinate_length
    # Check the total length (this checks the correct dimensionality)
    assert len(actual) == header_length + coordinate_length
    # Check the header
    assert actual[:header_length] == expected[:header_length]
    # Check the coordinates (using numpy.isnan; there are many byte representations for
    # NaN)
    assert np.isnan(struct.unpack("<3d", actual[header_length:])).all()


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="M coordinates not supported with GEOS < 3.12",
)
@pytest.mark.parametrize(
    "geom,expected",
    [
        pytest.param(empty_point_m, POINTM_NAN_WKB, id="POINT M EMPTY"),
        pytest.param(
            multi_point_empty_m,
            MULTIPOINTM_NAN_WKB,
            id="MULTIPOINT M EMPTY",
        ),
        pytest.param(
            shapely.geometrycollections([empty_point_m]),
            GEOMETRYCOLLECTIONM_NAN_WKB,
            id="GEOMETRYCOLLECTION (POINT M EMPTY)",
        ),
        pytest.param(
            shapely.geometrycollections([multi_point_empty_m]),
            NESTED_COLLECTIONM_NAN_WKB,
            id="GEOMETRYCOLLECTION (MULTIPOINT M EMPTY)",
        ),
    ],
)
def test_to_wkb_point_empty_m(geom, expected):
    actual = shapely.to_wkb(geom, output_dimension=3, byte_order=1)
    # Split 'actual' into header and coordinates
    coordinate_length = 8 * 3
    header_length = len(expected) - coordinate_length
    assert len(actual) == header_length + coordinate_length
    assert actual[:header_length] == expected[:header_length]
    assert np.isnan(struct.unpack("<3d", actual[header_length:])).all()


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="M coordinates not supported with GEOS < 3.12",
)
@pytest.mark.parametrize(
    "geom,expected",
    [
        pytest.param(empty_point_zm, POINTZM_NAN_WKB, id="POINT ZM EMPTY"),
        pytest.param(
            multi_point_empty_zm,
            MULTIPOINTZM_NAN_WKB,
            id="MULTIPOINT ZM EMPTY",
        ),
        pytest.param(
            shapely.geometrycollections([empty_point_zm]),
            GEOMETRYCOLLECTIONZM_NAN_WKB,
            id="GEOMETRYCOLLECTION (POINT ZM EMPTY)",
        ),
        pytest.param(
            shapely.geometrycollections([multi_point_empty_zm]),
            NESTED_COLLECTIONZM_NAN_WKB,
            id="GEOMETRYCOLLECTION (MULTIPOINT ZM EMPTY)",
        ),
    ],
)
def test_to_wkb_point_empty_zm(geom, expected):
    actual = shapely.to_wkb(geom, output_dimension=4, byte_order=1)
    # Split 'actual' into header and coordinates
    coordinate_length = 8 * 4
    header_length = len(expected) - coordinate_length
    assert len(actual) == header_length + coordinate_length
    assert actual[:header_length] == expected[:header_length]
    assert np.isnan(struct.unpack("<4d", actual[header_length:])).all()


@pytest.mark.parametrize(
    "geom,expected",
    [
        pytest.param(empty_point, POINT_NAN_WKB, id="POINT EMPTY"),
        pytest.param(multi_point_empty, MULTIPOINT_NAN_WKB, id="MULTIPOINT EMPTY"),
        pytest.param(
            shapely.geometrycollections([empty_point]),
            GEOMETRYCOLLECTION_NAN_WKB,
            id="GEOMETRYCOLLECTION (POINT EMPTY)",
        ),
        pytest.param(
            shapely.geometrycollections([multi_point_empty]),
            NESTED_COLLECTION_NAN_WKB,
            id="GEOMETRYCOLLECTION (MULTIPOINT EMPTY)",
        ),
    ],
)
def test_to_wkb_point_empty_2d_output_dim_3(geom, expected):
    actual = shapely.to_wkb(geom, output_dimension=3, byte_order=1)
    # Split 'actual' into header and coordinates
    coordinate_length = 16
    header_length = len(expected) - coordinate_length
    # Check the total length (this checks the correct dimensionality)
    assert len(actual) == header_length + coordinate_length
    # Check the header
    assert actual[:header_length] == expected[:header_length]
    # Check the coordinates (using numpy.isnan; there are many byte representations for
    # NaN)
    assert np.isnan(struct.unpack("<2d", actual[header_length:])).all()


@pytest.mark.parametrize(
    "wkb,expected_type,expected_dim",
    [
        pytest.param(POINT_NAN_WKB, 0, 2, id="POINT_NAN_WKB"),
        pytest.param(POINTZ_NAN_WKB, 0, 3, id="POINTZ_NAN_WKB"),
        pytest.param(MULTIPOINT_NAN_WKB, 4, 2, id="MULTIPOINT_NAN_WKB"),
        pytest.param(MULTIPOINTZ_NAN_WKB, 4, 3, id="MULTIPOINTZ_NAN_WKB"),
        pytest.param(GEOMETRYCOLLECTION_NAN_WKB, 7, 2, id="GEOMETRYCOLLECTION_NAN_WKB"),
        pytest.param(
            GEOMETRYCOLLECTIONZ_NAN_WKB, 7, 3, id="GEOMETRYCOLLECTIONZ_NAN_WKB"
        ),
        pytest.param(NESTED_COLLECTION_NAN_WKB, 7, 2, id="NESTED_COLLECTION_NAN_WKB"),
        pytest.param(NESTED_COLLECTIONZ_NAN_WKB, 7, 3, id="NESTED_COLLECTIONZ_NAN_WKB"),
    ],
)
def test_from_wkb_point_empty(wkb, expected_type, expected_dim):
    geom = shapely.from_wkb(wkb)
    # POINT (nan nan) transforms to an empty point
    assert shapely.is_empty(geom)
    assert shapely.get_type_id(geom) == expected_type
    # The dimensionality (2D/3D) is only read correctly for GEOS >= 3.9.0
    if shapely.geos_version >= (3, 9, 0):
        assert shapely.get_coordinate_dimension(geom) == expected_dim


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="M coordinates not supported with GEOS < 3.12",
)
@pytest.mark.parametrize(
    "wkb,expected_type",
    [
        pytest.param(POINTM_NAN_WKB, 0, id="POINTM_NAN_WKB"),
        pytest.param(MULTIPOINTM_NAN_WKB, 4, id="MULTIPOINTM_NAN_WKB"),
        pytest.param(GEOMETRYCOLLECTIONM_NAN_WKB, 7, id="GEOMETRYCOLLECTIONM_NAN_WKB"),
        pytest.param(NESTED_COLLECTIONM_NAN_WKB, 7, id="NESTED_COLLECTIONM_NAN_WKB"),
    ],
)
def test_from_wkb_point_empty_m(wkb, expected_type):
    geom = shapely.from_wkb(wkb)

    assert shapely.is_empty(geom)
    assert shapely.get_type_id(geom) == expected_type
    assert shapely.get_coordinate_dimension(geom) == 3
    assert not shapely.has_z(geom)
    assert shapely.has_m(geom)


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="M coordinates not supported with GEOS < 3.12",
)
@pytest.mark.parametrize(
    "wkb,expected_type",
    [
        pytest.param(POINTZM_NAN_WKB, 0, id="POINTZM_NAN_WKB"),
        pytest.param(MULTIPOINTZM_NAN_WKB, 4, id="MULTIPOINTZM_NAN_WKB"),
        pytest.param(
            GEOMETRYCOLLECTIONZM_NAN_WKB, 7, id="GEOMETRYCOLLECTIONZM_NAN_WKB"
        ),
        pytest.param(NESTED_COLLECTIONZM_NAN_WKB, 7, id="NESTED_COLLECTIONZM_NAN_WKB"),
    ],
)
def test_from_wkb_point_empty_zm(wkb, expected_type):
    geom = shapely.from_wkb(wkb)

    assert shapely.is_empty(geom)
    assert shapely.get_type_id(geom) == expected_type
    assert shapely.get_coordinate_dimension(geom) == 4
    assert shapely.has_z(geom)
    assert shapely.has_m(geom)


def test_to_wkb_point_empty_srid():
    expected = shapely.set_srid(empty_point, 4236)
    wkb = shapely.to_wkb(expected, include_srid=True)
    actual = shapely.from_wkb(wkb)
    assert shapely.get_srid(actual) == 4236


@pytest.mark.parametrize("geom", all_types + (point_z, empty_point))
def test_pickle(geom):
    pickled = pickle.dumps(geom)
    assert_geometries_equal(pickle.loads(pickled), geom, tolerance=0)


@pytest.mark.parametrize("geom", all_types_z)
def test_pickle_z(geom):
    if shapely.get_type_id(geom) == shapely.GeometryType.LINEARRING:
        # https://github.com/shapely/shapely/pull/1808#discussion_r1182174077
        pytest.xfail("TODO: Z gets dropped for LinearRing")
    pickled = pickle.dumps(geom)
    actual = pickle.loads(pickled)
    assert_geometries_equal(actual, geom, tolerance=0)
    if not actual.is_empty:  # GEOSHasZ with EMPTY geometries is inconsistent
        assert actual.has_z
    if shapely.geos_version >= (3, 12, 0):
        assert not actual.has_m


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="M coordinates not supported with GEOS < 3.12",
)
@pytest.mark.parametrize("geom", all_types_m)
def test_pickle_m(geom):
    if shapely.get_type_id(geom) == shapely.GeometryType.LINEARRING:
        pytest.xfail("TODO: M gets dropped for LinearRing types; see GH-2005")
    pickled = pickle.dumps(geom)
    actual = pickle.loads(pickled)
    assert_geometries_equal(actual, geom, tolerance=0)
    assert not actual.has_z
    if not actual.is_empty:  # GEOSHasM with EMPTY geometries is inconsistent
        assert actual.has_m


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="M coordinates not supported with GEOS < 3.12",
)
@pytest.mark.parametrize("geom", all_types_zm)
def test_pickle_zm(geom):
    if shapely.get_type_id(geom) == shapely.GeometryType.LINEARRING:
        pytest.xfail("TODO: ZM gets dropped for LinearRing types; see GH-2005")
    pickled = pickle.dumps(geom)
    actual = pickle.loads(pickled)
    assert_geometries_equal(actual, geom, tolerance=0)
    if not actual.is_empty:  # GEOSHasZ with EMPTY geometries is inconsistent
        assert actual.has_z
        assert actual.has_m


@pytest.mark.parametrize("geom", all_types + (point_z, empty_point))
def test_pickle_with_srid(geom):
    geom = shapely.set_srid(geom, 4326)
    pickled = pickle.dumps(geom)
    assert shapely.get_srid(pickle.loads(pickled)) == 4326


@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
@pytest.mark.parametrize(
    "geojson,expected",
    [
        pytest.param(
            GEOJSON_GEOMETRY, GEOJSON_GEOMETRY_EXPECTED, id="GEOJSON_GEOMETRY"
        ),
        pytest.param(GEOJSON_FEATURE, GEOJSON_GEOMETRY_EXPECTED, id="GEOJSON_FEATURE"),
        pytest.param(
            GEOJSON_FEATURECOLECTION,
            shapely.geometrycollections(GEOJSON_COLLECTION_EXPECTED),
            id="GEOJSON_FEATURECOLECTION",
        ),
        pytest.param(
            [GEOJSON_GEOMETRY] * 2,
            [GEOJSON_GEOMETRY_EXPECTED] * 2,
            id="GEOJSON_GEOMETRYx2",
        ),
        pytest.param(None, None, id="None"),
        pytest.param(
            [GEOJSON_GEOMETRY, None],
            [GEOJSON_GEOMETRY_EXPECTED, None],
            id="GEOJSON_GEOMETRY_None",
        ),
    ],
)
def test_from_geojson(geojson, expected):
    actual = shapely.from_geojson(geojson)
    assert_geometries_equal(actual, expected)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
def test_from_geojson_exceptions():
    with pytest.raises(TypeError, match="Expected bytes or string, got int"):
        shapely.from_geojson(1)

    with pytest.raises(shapely.GEOSException, match="Error parsing JSON"):
        shapely.from_geojson("")

    with pytest.raises(shapely.GEOSException, match="Unknown geometry type"):
        shapely.from_geojson('{"type": "NoGeometry", "coordinates": []}')

    with pytest.raises(shapely.GEOSException, match="type must be array, but is null"):
        shapely.from_geojson('{"type": "LineString", "coordinates": null}')

    # Note: The two below tests are the reason that from_geojson is disabled for
    # GEOS 3.10.0 See https://trac.osgeo.org/geos/ticket/1138
    with pytest.raises(shapely.GEOSException, match="key 'type' not found"):
        shapely.from_geojson('{"geometry": null, "properties": []}')

    with pytest.raises(shapely.GEOSException, match="key 'type' not found"):
        shapely.from_geojson('{"no": "geojson"}')


@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
def test_from_geojson_warn_on_invalid():
    with pytest.warns(Warning, match="Invalid GeoJSON"):
        assert shapely.from_geojson("", on_invalid="warn") is None


@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
def test_from_geojson_ignore_on_invalid():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert shapely.from_geojson("", on_invalid="ignore") is None


@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
def test_from_geojson_on_invalid_unsupported_option():
    with pytest.raises(ValueError, match="not a valid option"):
        shapely.from_geojson(GEOJSON_GEOMETRY, on_invalid="unsupported_option")


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "expected,geometry",
    [
        pytest.param(
            GEOJSON_GEOMETRY, GEOJSON_GEOMETRY_EXPECTED, id="GEOJSON_GEOMETRY"
        ),
        pytest.param(
            [GEOJSON_GEOMETRY] * 2,
            [GEOJSON_GEOMETRY_EXPECTED] * 2,
            id="GEOJSON_GEOMETRYx2",
        ),
        pytest.param(None, None, id="None"),
        pytest.param(
            [GEOJSON_GEOMETRY, None],
            [GEOJSON_GEOMETRY_EXPECTED, None],
            id="GEOJSON_GEOMETRY_None",
        ),
    ],
)
def test_to_geojson(geometry, expected):
    actual = shapely.to_geojson(geometry, indent=4)
    assert np.all(actual == np.asarray(expected))


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize("indent", [None, 0, 4])
def test_to_geojson_indent(indent):
    separators = (",", ":") if indent is None else (",", ": ")
    expected = json.dumps(
        json.loads(GEOJSON_GEOMETRY), indent=indent, separators=separators
    )
    actual = shapely.to_geojson(GEOJSON_GEOMETRY_EXPECTED, indent=indent)
    assert actual == expected


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
def test_to_geojson_exceptions():
    with pytest.raises(TypeError):
        shapely.to_geojson(1)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geom",
    [
        empty_point,
        shapely.multipoints([empty_point, point]),
        shapely.geometrycollections([empty_point, point]),
        shapely.geometrycollections(
            [shapely.geometrycollections([empty_point]), point]
        ),
    ],
)
def test_to_geojson_point_empty(geom):
    # Pending GEOS ticket: https://trac.osgeo.org/geos/ticket/1139
    with pytest.raises(ValueError):
        assert shapely.to_geojson(geom)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason="GEOS < 3.10.1")
@pytest.mark.parametrize("geom", all_types)
def test_geojson_all_types(geom):
    type_id = shapely.get_type_id(geom)
    if type_id == shapely.GeometryType.LINEARRING:
        pytest.skip("Linearrings are not preserved in GeoJSON")
    elif geom.is_empty and (
        type_id == shapely.GeometryType.POINT
        or (
            type_id == shapely.GeometryType.MULTIPOINT
            and shapely.get_num_geometries(geom) > 0
        )
    ):
        with pytest.raises(ValueError):  # same as test_to_geojson_point_empty
            assert shapely.to_geojson(geom)
        return
    geojson = shapely.to_geojson(geom)
    actual = shapely.from_geojson(geojson)
    assert not actual.has_z
    geoms_are_empty = shapely.is_empty([geom, actual])
    if geoms_are_empty.any():
        # Ensure both are EMPTY
        assert geoms_are_empty.all()
    else:
        assert_geometries_equal(actual, geom)
