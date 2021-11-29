import numpy as np
import pytest

import pygeos
from pygeos.testing import assert_geometries_equal

from .common import (
    empty_line_string,
    empty_point,
    line_string,
    linear_ring,
    multi_line_string,
    multi_point,
    multi_polygon,
    point,
    polygon,
)


def test_line_interpolate_point_geom_array():
    actual = pygeos.line_interpolate_point(
        [line_string, linear_ring, multi_line_string], -1
    )
    assert_geometries_equal(actual[0], pygeos.Geometry("POINT (1 0)"))
    assert_geometries_equal(actual[1], pygeos.Geometry("POINT (0 1)"))
    assert_geometries_equal(
        actual[2], pygeos.Geometry("POINT (0.5528 1.1056)"), tolerance=0.001
    )


def test_line_interpolate_point_geom_array_normalized():
    actual = pygeos.line_interpolate_point(
        [line_string, linear_ring, multi_line_string], 1, normalized=True
    )
    assert_geometries_equal(actual[0], pygeos.Geometry("POINT (1 1)"))
    assert_geometries_equal(actual[1], pygeos.Geometry("POINT (0 0)"))
    assert_geometries_equal(actual[2], pygeos.Geometry("POINT (1 2)"))


def test_line_interpolate_point_float_array():
    actual = pygeos.line_interpolate_point(line_string, [0.2, 1.5, -0.2])
    assert_geometries_equal(actual[0], pygeos.Geometry("POINT (0.2 0)"))
    assert_geometries_equal(actual[1], pygeos.Geometry("POINT (1 0.5)"))
    assert_geometries_equal(actual[2], pygeos.Geometry("POINT (1 0.8)"))


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize(
    "geom",
    [
        pygeos.Geometry("LINESTRING EMPTY"),
        pygeos.Geometry("LINEARRING EMPTY"),
        pygeos.Geometry("MULTILINESTRING EMPTY"),
        pygeos.Geometry("MULTILINESTRING (EMPTY, (0 0, 1 1))"),
        pygeos.Geometry("GEOMETRYCOLLECTION EMPTY"),
        pygeos.Geometry("GEOMETRYCOLLECTION (LINESTRING EMPTY, POINT (1 1))"),
    ],
)
def test_line_interpolate_point_empty(geom, normalized):
    # These geometries segfault in some versions of GEOS (in 3.8.0, still
    # some of them segfault). Instead, we patched this to return POINT EMPTY.
    # This matches GEOS 3.8.0 behavior on simple empty geometries.
    assert_geometries_equal(
        pygeos.line_interpolate_point(geom, 0.2, normalized=normalized), empty_point
    )


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize(
    "geom",
    [
        empty_point,
        point,
        polygon,
        multi_point,
        multi_polygon,
        pygeos.geometrycollections([point]),
        pygeos.geometrycollections([polygon]),
        pygeos.geometrycollections([multi_line_string]),
        pygeos.geometrycollections([multi_point]),
        pygeos.geometrycollections([multi_polygon]),
    ],
)
def test_line_interpolate_point_invalid_type(geom, normalized):
    with pytest.raises(TypeError):
        assert pygeos.line_interpolate_point(geom, 0.2, normalized=normalized)


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


@pytest.mark.parametrize("normalized", [False, True])
def test_line_locate_point_none(normalized):
    assert np.isnan(pygeos.line_locate_point(line_string, None, normalized=normalized))
    assert np.isnan(pygeos.line_locate_point(None, point, normalized=normalized))


@pytest.mark.parametrize("normalized", [False, True])
def test_line_locate_point_empty(normalized):
    assert np.isnan(
        pygeos.line_locate_point(line_string, empty_point, normalized=normalized)
    )
    assert np.isnan(
        pygeos.line_locate_point(empty_line_string, point, normalized=normalized)
    )


@pytest.mark.parametrize("normalized", [False, True])
def test_line_locate_point_invalid_geometry(normalized):
    with pytest.raises(pygeos.GEOSException):
        pygeos.line_locate_point(line_string, line_string, normalized=normalized)

    with pytest.raises(pygeos.GEOSException):
        pygeos.line_locate_point(polygon, point, normalized=normalized)


def test_line_merge_geom_array():
    actual = pygeos.line_merge([line_string, multi_line_string])
    assert_geometries_equal(actual[0], line_string)
    assert_geometries_equal(actual[1], pygeos.Geometry("LINESTRING (0 0, 1 2)"))


def test_shared_paths_linestring():
    g1 = pygeos.linestrings([(0, 0), (1, 0), (1, 1)])
    g2 = pygeos.linestrings([(0, 0), (1, 0)])
    actual1 = pygeos.shared_paths(g1, g2)
    assert_geometries_equal(
        pygeos.get_geometry(actual1, 0), pygeos.multilinestrings([g2])
    )


def test_shared_paths_none():
    assert pygeos.shared_paths(line_string, None) is None
    assert pygeos.shared_paths(None, line_string) is None
    assert pygeos.shared_paths(None, None) is None


def test_shared_paths_non_linestring():
    g1 = pygeos.linestrings([(0, 0), (1, 0), (1, 1)])
    g2 = pygeos.points(0, 1)
    with pytest.raises(pygeos.GEOSException):
        pygeos.shared_paths(g1, g2)


def _prepare_input(geometry, prepare):
    """Prepare without modifying inplace"""
    if prepare:
        geometry = pygeos.apply(geometry, lambda x: x)  # makes a copy
        pygeos.prepare(geometry)
        return geometry
    else:
        return geometry


@pytest.mark.parametrize("prepare", [True, False])
def test_shortest_line(prepare):
    g1 = pygeos.linestrings([(0, 0), (1, 0), (1, 1)])
    g2 = pygeos.linestrings([(0, 3), (3, 0)])
    actual = pygeos.shortest_line(_prepare_input(g1, prepare), g2)
    expected = pygeos.linestrings([(1, 1), (1.5, 1.5)])
    assert pygeos.equals(actual, expected)


@pytest.mark.parametrize("prepare", [True, False])
def test_shortest_line_none(prepare):
    assert pygeos.shortest_line(_prepare_input(line_string, prepare), None) is None
    assert pygeos.shortest_line(None, line_string) is None
    assert pygeos.shortest_line(None, None) is None


@pytest.mark.parametrize("prepare", [True, False])
def test_shortest_line_empty(prepare):
    g1 = _prepare_input(line_string, prepare)
    assert pygeos.shortest_line(g1, empty_line_string) is None
    g1_empty = _prepare_input(empty_line_string, prepare)
    assert pygeos.shortest_line(g1_empty, line_string) is None
    assert pygeos.shortest_line(g1_empty, empty_line_string) is None
