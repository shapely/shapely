import pygeos
import numpy as np
import pytest

from pygeos import Geometry

from .common import polygon
from .common import point


def test_buffer_default():
    # buffer a point to a circle
    radii = np.array([1.0, 2.0])
    actual = pygeos.buffer(point, radii, quadsegs=16)
    assert pygeos.area(actual) == pytest.approx(np.pi * radii ** 2, rel=0.01)


def test_buffer_square():
    # buffer a point to a square
    actual = pygeos.buffer(point, 1.0, cap_style="square")
    assert pygeos.area(actual) == pytest.approx(2 ** 2, abs=0.01)


def test_buffer_single_sided():
    # buffer a line on one side
    line = pygeos.linestrings([[0, 0], [10, 0]])
    actual = pygeos.buffer(line, 0.1, cap_style="square", single_sided=True)
    assert pygeos.area(actual) == pytest.approx(0.1 * 10, abs=0.01)


def test_delaunay_triangles():
    original = Geometry("MULTIPOINT (50 30, 60 30, 100 100, 10 150, 110 120)")
    actual = pygeos.delaunay_triangles(original)
    assert pygeos.get_num_geometries(actual) == 3


def test_delaunay_triangles_only_edges():
    original = Geometry("MULTIPOINT (50 30, 60 30, 100 100, 10 150, 110 120)")
    actual = pygeos.delaunay_triangles(original, only_edges=True)
    assert pygeos.get_num_geometries(actual) == 7


def test_centroid():
    actual = pygeos.centroid(polygon)
    assert pygeos.equals(actual, pygeos.points(1, 1))


def test_simplify():
    line = pygeos.linestrings([[0, 0], [0.1, 1], [0, 2]])
    actual = pygeos.simplify(line, [0, 1.0])
    assert pygeos.get_num_points(actual).tolist() == [3, 2]


def test_simplify_nan():
    actual = pygeos.simplify(
        np.array([point, np.nan, np.nan, None, point]),
        np.array([np.nan, 1.0, np.nan, 1.0, 1.0]),
    )
    assert pygeos.equals(actual[-1], point)
    assert pygeos.is_empty(actual[:-1]).all()


def test_snap():
    line = pygeos.linestrings([[0, 0], [1, 0], [2, 0]])
    points = pygeos.points([0, 1], [1, 0.1])
    actual = pygeos.snap(points, line, 0.5)
    expected = pygeos.points([0, 1], [1, 0])
    assert pygeos.equals(actual, expected).all()


def test_voronoi_polygons():
    original = Geometry("MULTIPOINT (50 30, 60 30, 100 100, 10 150, 110 120)")
    actual = pygeos.voronoi_polygons(original)
    assert pygeos.get_num_geometries(actual) == 5


def test_voronoi_polygons_only_edges():
    # example from PostGIS docs
    original = Geometry("MULTIPOINT (50 30, 60 30, 100 100, 10 150, 110 120)")
    actual = pygeos.voronoi_polygons(original, only_edges=True)
    assert pygeos.get_num_geometries(actual) == 7
