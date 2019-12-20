import pygeos
import numpy as np
import pytest

from .common import empty_point
from .common import empty_line_string
from .common import point
from .common import line_string
from .common import linear_ring
from .common import multi_line_string


def test_line_interpolate_point_geom_array():
    actual = pygeos.line_interpolate_point([line_string, linear_ring], -1)
    assert pygeos.equals(actual[0], pygeos.Geometry("POINT (1 0)"))
    assert pygeos.equals(actual[1], pygeos.Geometry("POINT (0 1)"))


def test_line_interpolate_point_float_array():
    actual = pygeos.line_interpolate_point(line_string, [0.2, 1.5, -0.2])
    assert pygeos.equals(actual[0], pygeos.Geometry("POINT (0.2 0)"))
    assert pygeos.equals(actual[1], pygeos.Geometry("POINT (1 0.5)"))
    assert pygeos.equals(actual[2], pygeos.Geometry("POINT (1 0.8)"))


def test_line_interpolate_point_empty():
    assert pygeos.equals(
        pygeos.line_interpolate_point(empty_line_string, 0.2), empty_point
    )


def test_line_interpolate_point_none():
    assert pygeos.line_interpolate_point(None, 0.2) is None


def test_line_interpolate_point_nan():
    assert pygeos.line_interpolate_point(line_string, np.nan) is None


def test_line_locate_point_geom_array():
    point = pygeos.points(0, 1)
    actual = pygeos.line_locate_point([line_string, linear_ring], point)
    np.testing.assert_allclose(actual, [0.0, 3.0])


def test_line_locate_point_geom_array2():
    points = pygeos.points([[0, 0], [1, 0]])
    actual = pygeos.line_locate_point(line_string, points)
    np.testing.assert_allclose(actual, [0.0, 1.0])


def test_line_locate_point_none():
    assert np.isnan(pygeos.line_locate_point(line_string, None))
    assert np.isnan(pygeos.line_locate_point(None, point))


def test_line_locate_point_empty():
    assert np.isnan(pygeos.line_locate_point(line_string, empty_point))
    assert np.isnan(pygeos.line_locate_point(empty_line_string, point))


def test_line_merge_geom_array():
    actual = pygeos.line_merge([line_string, multi_line_string])
    assert pygeos.equals(actual[0], line_string)
    assert pygeos.equals(actual[1], multi_line_string)


def test_shared_paths_linestring():
    g1 = pygeos.linestrings([(0, 0), (1, 0), (1, 1)])
    g2 = pygeos.linestrings([(0, 0), (1, 0)])
    actual1 = pygeos.shared_paths(g1, g2)
    assert pygeos.equals(pygeos.get_geometry(actual1, 0), g2)


def test_shared_paths_none():
    assert pygeos.shared_paths(line_string, None) is None
    assert pygeos.shared_paths(None, line_string) is None
    assert pygeos.shared_paths(None, None) is None


def test_shared_paths_non_linestring():
    g1 = pygeos.linestrings([(0, 0), (1, 0), (1, 1)])
    g2 = pygeos.points(0, 1)
    with pytest.raises(pygeos.GEOSException):
        pygeos.shared_paths(g1, g2)
