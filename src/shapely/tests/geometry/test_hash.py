import pytest

import shapely
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point


@pytest.mark.parametrize(
    "geom",
    [
        Point(1, 2),
        MultiPoint([(1, 2), (3, 4)]),
        LineString([(1, 2), (3, 4)]),
        Point(0, 0).buffer(1.0),
        GeometryCollection([Point(1, 2), LineString([(1, 2), (3, 4)])]),
    ],
    ids=[
        "Point",
        "MultiPoint",
        "LineString",
        "Polygon",
        "GeometryCollection",
    ],
)
def test_hash(geom):
    h1 = hash(geom)
    assert h1 == hash(shapely.from_wkb(geom.wkb))
    assert h1 != hash(translate(geom, 1.0, 2.0))


@pytest.mark.parametrize(
    "wkt",
    [
        "POINT (0 0)",
        "LINESTRING (0 0, 1 0)",
        "LINEARRING (0 0, 1 0, 1 1, 0 0)",
        "POLYGON ((0 0, 3 0, 3 3, 0 0), (1 0, 2 0, 2 1, 1 0))",
        "MULTIPOINT ((0 0), (1 0))",
        "MULTILINESTRING ((0 0, 1 0), (0 1, 1 1))",
        "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)))",
        (
            "GEOMETRYCOLLECTION (POINT (0 0), GEOMETRYCOLLECTION "
            "(LINESTRING (0 0, 1 0), POINT EMPTY))"
        ),
        "POINT Z (1 2 0)",
        pytest.param(
            "POINT M (1 2 0)",
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12"
            ),
        ),
        pytest.param(
            "POINT ZM (1 2 0 0)",
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 12, 0), reason="GEOS < 3.12"
            ),
        ),
    ],
)
def test_hash_signed_zero(wkt):
    left = shapely.from_wkt(wkt)
    right = shapely.from_wkt(wkt.replace("0", "-0"))
    assert left == right
    assert hash(left) == hash(right)
    assert {left: "value"}[right] == "value"
    assert len({left, right}) == 1


@pytest.mark.parametrize("srid", [0, 4326, -1])
@pytest.mark.parametrize(
    "wkt",
    [
        "POINT (1 2)",
        "LINESTRING (1 2, 3 4)",
        "POLYGON ((0 0, 1 0, 1 1, 0 0))",
        "GEOMETRYCOLLECTION (POINT (1 2), GEOMETRYCOLLECTION (POINT EMPTY))",
        "POINT EMPTY",
        "POLYGON EMPTY",
        "GEOMETRYCOLLECTION EMPTY",
    ],
)
def test_hash_ignores_srid(wkt, srid):
    left = shapely.from_wkt(wkt)
    right = shapely.set_srid(left, srid)
    assert left == right
    assert hash(left) == hash(right)
    assert {left: "value"}[right] == "value"
    assert len({left, right}) == 1


@pytest.mark.parametrize("bits", [0x7FF8000000000001, 0xFFF8000000000002])
@pytest.mark.parametrize(
    "dimension, ordinate",
    [(dim, i) for dim in ["", "Z", "M", "ZM"] for i in range(2 + len(dim))],
)
@pytest.mark.parametrize("nested", [False, True])
def test_hash_nan_payload(bits, dimension, ordinate, nested):
    import struct

    if "M" in dimension and shapely.geos_version < (3, 12, 0):
        pytest.skip("GEOS < 3.12")
    nan = struct.unpack("d", struct.pack("Q", bits))[0]
    ndim = 2 + len(dimension)
    type_id = 1 | (0x80000000 if "Z" in dimension else 0)
    type_id |= 0x40000000 if "M" in dimension else 0

    def point(value):
        coords = [1.0] * ndim
        coords[ordinate] = value
        return shapely.from_wkb(struct.pack("<BI" + "d" * ndim, 1, type_id, *coords))

    left, right = point(float("nan")), point(nan)
    if nested:
        left = GeometryCollection([GeometryCollection([left])])
        right = GeometryCollection([GeometryCollection([right])])
    assert left == right
    assert hash(left) == hash(right)
    assert {left: "value"}[right] == "value"
    assert len({left, right}) == 1


@pytest.mark.parametrize(
    "wkt",
    [
        "POINT",
        "LINESTRING",
        "POLYGON",
        "MULTIPOINT",
        "MULTILINESTRING",
        "MULTIPOLYGON",
        "GEOMETRYCOLLECTION",
    ],
)
def test_hash_empty_dimensions(wkt):
    dimensions = (
        ["", "Z", "M", "ZM"] if shapely.geos_version >= (3, 12, 0) else ["", "Z"]
    )
    geometries = [shapely.from_wkt(f"{wkt} {dim} EMPTY") for dim in dimensions]
    for left in geometries:
        for right in geometries:
            if left == right:
                assert hash(left) == hash(right)
                assert {left: "value"}[right] == "value"


def test_hash_nan_z_equality():
    left = LineString([(0, 1, float("nan")), (2, 3, float("nan"))])
    right = LineString([(0, 1, float("nan")), (2, 3, 4)])
    if left == right:
        assert hash(left) == hash(right)
        assert {left: "value"}[right] == "value"
    point = Point(0, 1, float("nan"))
    if point == Point(0, 1):
        assert hash(point) == hash(Point(0, 1))


def test_hash_does_not_mutate_geometry():
    import pickle

    geom = shapely.set_srid(
        GeometryCollection([Point(-0.0, 1), Point(1, float("nan"))]), 4326
    )
    before_wkb = shapely.to_wkb(geom, include_srid=True)
    before_pickle = pickle.dumps(geom)
    assert hash(geom) == hash(geom)
    assert shapely.to_wkb(geom, include_srid=True) == before_wkb
    assert pickle.dumps(geom) == before_pickle
    assert shapely.get_srid(geom) == 4326


def test_hash_preserves_structural_equality():
    line = LineString([(0, 0), (1, 1)])
    reverse = LineString([(1, 1), (0, 0)])
    assert line.equals(reverse)
    assert line != reverse
    assert len({line, reverse}) == 2


@pytest.mark.skipif(shapely.geos_version < (3, 13, 0), reason="GEOS < 3.13")
def test_hash_nonlinear_collection():
    # Nonlinear children can be parsed, but are not supported by Shapely.
    geom = shapely.from_wkt("GEOMETRYCOLLECTION (CIRCULARSTRING (0 0, 1 1, 2 0))")
    with pytest.raises(NotImplementedError, match="Nonlinear"):
        hash(geom)
