import platform
import weakref

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
from shapely.errors import ShapelyDeprecationWarning


def test_polygon():
    assert bool(Polygon()) is False


def test_linestring():
    assert bool(LineString()) is False


def test_point():
    assert bool(Point()) is False


def test_geometry_collection():
    assert bool(GeometryCollection()) is False


geometries_all_types = [
    Point(1, 1),
    LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)]),
    LineString([(0, 0), (1, 1), (0, 1), (0, 0)]),
    Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
    MultiPoint([(1, 1)]),
    MultiLineString([[(0, 0), (1, 1), (0, 1), (0, 0)]]),
    MultiPolygon([Polygon([(0, 0), (1, 1), (0, 1), (0, 0)])]),
    GeometryCollection([Point(1, 1)]),
]


@pytest.mark.skipif(
    platform.python_implementation() == "PyPy",
    reason="Setting custom attributes doesn't fail on PyPy",
)
@pytest.mark.parametrize("geom", geometries_all_types)
def test_setattr_disallowed(geom):
    with pytest.raises(AttributeError):
        geom.name = "test"


@pytest.mark.parametrize("geom", geometries_all_types)
def test_weakrefable(geom):
    _ = weakref.ref(geom)


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


def test_base_class_not_callable():
    with pytest.raises(TypeError):
        shapely.Geometry("POINT (1 1)")


def test_GeometryType_deprecated():
    geom = Point(1, 1)

    with pytest.warns(ShapelyDeprecationWarning):
        geom_type = geom.geometryType()

    assert geom_type == geom.geom_type


def test_type_deprecated():
    geom = Point(1, 1)

    with pytest.warns(ShapelyDeprecationWarning):
        geom_type = geom.type

    assert geom_type == geom.geom_type


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
def test_segmentize():
    line = LineString([(0, 0), (0, 10)])
    result = line.segmentize(max_segment_length=5)
    assert result.equals(LineString([(0, 0), (0, 5), (0, 10)]))


@pytest.mark.skipif(shapely.geos_version < (3, 7, 0), reason="GEOS < 3.7")
def test_reverse():
    coords = [(0, 0), (1, 2)]
    line = LineString(coords)
    result = line.reverse()
    assert result.coords[:] == coords[::-1]


@pytest.mark.skipif(shapely.geos_version < (3, 9, 0), reason="GEOS < 3.9")
@pytest.mark.parametrize(
    "op", ["union", "intersection", "difference", "symmetric_difference"]
)
@pytest.mark.parametrize("grid_size", [0, 1, 2])
def test_binary_op_grid_size(op, grid_size):
    geom1 = shapely.box(0, 0, 2.5, 2.5)
    geom2 = shapely.box(2, 2, 3, 3)

    result = getattr(geom1, op)(geom2, grid_size=grid_size)
    expected = getattr(shapely, op)(geom1, geom2, grid_size=grid_size)
    assert result == expected


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
def test_dwithin():
    point = Point(1, 1)
    line = LineString([(0, 0), (0, 10)])
    assert point.dwithin(line, 0.5) is False
    assert point.dwithin(line, 1.5) is True


@pytest.mark.parametrize(
    "op", ["convex_hull", "envelope", "oriented_envelope", "minimum_rotated_rectangle"]
)
def test_constructive_properties(op):
    geom = LineString([(0, 0), (0, 10), (10, 10)])
    result = getattr(geom, op)
    expected = getattr(shapely, op)(geom)
    assert result == expected
