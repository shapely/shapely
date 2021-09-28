from shapely import geometry
from shapely.errors import ShapelyDeprecationWarning

import pytest


def test_polygon():
    assert bool(geometry.Polygon()) is False


def test_linestring():
    assert bool(geometry.LineString()) is False


def test_point():
    assert bool(geometry.Point()) is False


def test_geometry_collection():
    assert bool(geometry.GeometryCollection()) is False


@pytest.mark.parametrize("geom", [
    geometry.Point(1, 1),
    geometry.LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)]),
    geometry.LineString([(0, 0), (1, 1), (0, 1), (0, 0)]),
    geometry.Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
    geometry.MultiPoint([(1, 1)]),
    geometry.MultiLineString([[(0, 0), (1, 1), (0, 1), (0, 0)]]),
    geometry.MultiPolygon([geometry.Polygon([(0, 0), (1, 1), (0, 1), (0, 0)])]),
    geometry.GeometryCollection([geometry.Point(1, 1)]),
])
def test_setattr_disallowed(geom):
    with pytest.warns(ShapelyDeprecationWarning, match="Setting custom attributes"):
        geom.name = "test"
    assert geom.name == "test"
