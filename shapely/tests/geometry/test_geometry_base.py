import numpy as np
import pytest

from shapely import geometry


def test_polygon():
    assert bool(geometry.Polygon()) is False


def test_linestring():
    assert bool(geometry.LineString()) is False


def test_point():
    assert bool(geometry.Point()) is False


def test_geometry_collection():
    assert bool(geometry.GeometryCollection()) is False


geometries_all_types = [
    geometry.Point(1, 1),
    geometry.LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)]),
    geometry.LineString([(0, 0), (1, 1), (0, 1), (0, 0)]),
    geometry.Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
    geometry.MultiPoint([(1, 1)]),
    geometry.MultiLineString([[(0, 0), (1, 1), (0, 1), (0, 0)]]),
    geometry.MultiPolygon([geometry.Polygon([(0, 0), (1, 1), (0, 1), (0, 0)])]),
    geometry.GeometryCollection([geometry.Point(1, 1)]),
]


@pytest.mark.parametrize("geom", geometries_all_types)
def test_setattr_disallowed(geom):
    with pytest.raises(AttributeError):
        geom.name = "test"


@pytest.mark.parametrize("geom", geometries_all_types)
def test_comparison_notimplemented(geom):
    # comparing to a non-geometry class should return NotImplemented in __eq__
    # to ensure proper delegation to other (eg to ensure comparison of scalar
    # with array works)
    # https://github.com/shapely/shapely/issues/1056
    assert geom.__eq__(1) is NotImplemented

    # with array
    arr = np.array([geom, geom], dtype=object)

    result = arr == geom
    assert isinstance(result, np.ndarray)
    assert result.all()

    result = geom == arr
    assert isinstance(result, np.ndarray)
    assert result.all()

    result = arr != geom
    assert isinstance(result, np.ndarray)
    assert not result.any()

    result = geom != arr
    assert isinstance(result, np.ndarray)
    assert not result.any()
