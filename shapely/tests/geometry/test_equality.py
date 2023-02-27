import numpy as np
import pytest

import shapely
from shapely import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.tests.common import all_types, all_types_z


@pytest.mark.parametrize("geom", all_types + all_types_z)
def test_equality(geom):
    assert geom == geom


@pytest.mark.parametrize(
    "left, right",
    [
        (LineString([(0, 0), (1, 1)]), LineString([(0, 0), (1, 2)])),
        (LineString([(0, 0), (1, 1)]), LineString([(0, 0), (1, 1 + 1e-12)])),
        (LineString([(0, 0), (1, 1)]), LineString([(1, 1), (0, 0)])),
        # different order of sub-geometries
        (
            MultiLineString([[(1, 1), (2, 2)], [(2, 2), (3, 3)]]),
            MultiLineString([[(2, 2), (3, 3)], [(1, 1), (2, 2)]]),
        ),
    ],
)
def test_equality_false(left, right):
    assert left != right


@pytest.mark.parametrize(
    "left, right",
    [
        (LineString([(0, 1), (2, np.nan)]), LineString([(0, 1), (2, np.nan)])),
        (
            LineString([(0, 1), (np.nan, np.nan)]),
            LineString([(0, 1), (np.nan, np.nan)]),
        ),
        (LineString([(np.nan, 1), (2, 3)]), LineString([(np.nan, 1), (2, 3)])),
        (LineString([(0, np.nan), (2, 3)]), LineString([(0, np.nan), (2, 3)])),
        # NaN as explicit Z coordinate
        (
            LineString([(0, 1, np.nan), (2, 3, np.nan)]),
            LineString([(0, 1, np.nan), (2, 3, np.nan)]),
        ),
    ],
)
def test_equality_with_nan(left, right):
    assert left == right


@pytest.mark.parametrize(
    "left, right",
    [
        (LineString([(0, 1), (2, np.nan)]), LineString([(0, 1), (2, 3)])),
        (
            LineString([(0, 1, np.nan), (2, 3, np.nan)]),
            LineString([(0, 1, np.nan), (2, 3, 4)]),
        ),
    ],
)
def test_equality_with_nan_false(left, right):
    assert left != right


def test_equality_z():
    geom1 = Point(0, 1)
    geom2 = Point(0, 1, 2)
    assert geom1 != geom2

    geom2 = Point(0, 1, np.nan)
    if shapely.geos_version < (3, 12, 0):
        # older GEOS versions ignore NaN for Z also when explicitly created with 3D
        assert geom1 == geom2
    else:
        assert geom1 != geom2


def test_equality_exact_type():
    # geometries with different type but same coord seq are not equal
    geom1 = LineString([(0, 0), (1, 1), (0, 1), (0, 0)])
    geom2 = LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)])
    geom3 = Polygon([(0, 0), (1, 1), (0, 1), (0, 0)])
    assert geom1 != geom2
    assert geom1 != geom3
    assert geom2 != geom3

    # empty with different type
    geom1 = shapely.from_wkt("POINT EMPTY")
    geom2 = shapely.from_wkt("LINESTRING EMPTY")
    assert geom1 != geom2


def test_equality_polygon():
    # different exterior rings
    geom1 = shapely.from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")
    geom2 = shapely.from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 15, 0 0))")
    assert geom1 != geom2

    # different number of holes
    geom1 = shapely.from_wkt(
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1))"
    )
    geom2 = shapely.from_wkt(
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1), (3 3, 4 3, 4 4, 3 3))"
    )
    assert geom1 != geom2

    # different order of holes
    geom1 = shapely.from_wkt(
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 4 3, 4 4, 3 3), (1 1, 2 1, 2 2, 1 1))"
    )
    geom2 = shapely.from_wkt(
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 1), (3 3, 4 3, 4 4, 3 3))"
    )
    assert geom1 != geom2
