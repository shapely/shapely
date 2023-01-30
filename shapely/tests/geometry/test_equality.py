import numpy as np
import pytest

import shapely
from shapely import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.tests.common import all_types, all_types_z


@pytest.mark.parametrize("geom", all_types + all_types_z)
def test_equality(geom):
    assert geom == geom


def test_equality_false():
    geom1 = LineString([(0, 0), (1, 1)])
    geom2 = LineString([(0, 0), (1, 2)])
    assert geom1 != geom2


def test_equality_with_nan():
    geom1 = LineString([(0, 1), (3, np.nan)])
    geom2 = LineString([(0, 1), (3, np.nan)])
    assert geom1 == geom2

    geom1 = LineString([(0, 1, np.nan), (3, 4, np.nan)])
    geom2 = LineString([(0, 1, np.nan), (3, 4, np.nan)])
    assert geom1 == geom2


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

    # emtpy with different type
    geom1 = shapely.from_wkt("POINT EMPTY")
    geom2 = shapely.from_wkt("LINESTRING EMPTY")
    assert geom1 != geom2


def test_equality_different_order():
    geom1 = MultiLineString([[(1, 1), (2, 2)], [(2, 2), (3, 3)]])
    geom2 = MultiLineString([[(2, 2), (3, 3)], [(1, 1), (2, 2)]])
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
