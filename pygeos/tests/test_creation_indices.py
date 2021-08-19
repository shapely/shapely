import numpy as np
import pytest

import pygeos

from .common import assert_geometries_equal, line_string, linear_ring, point, polygon

pnts = pygeos.points
lstrs = pygeos.linestrings
geom_coll = pygeos.geometrycollections


@pytest.mark.parametrize(
    "func", [pygeos.points, pygeos.linestrings, pygeos.linearrings]
)
@pytest.mark.parametrize(
    "coordinates",
    [
        np.empty((2,)),  # not enough dimensions
        np.empty((2, 4, 1)),  # too many dimensions
        np.empty((2, 4)),  # wrong inner dimension size
        None,
        np.full((2, 2), "foo", dtype=object),  # wrong type
    ],
)
def test_invalid_coordinates(func, coordinates):
    with pytest.raises((TypeError, ValueError)):
        func(coordinates, indices=[0, 1])


@pytest.mark.parametrize(
    "func",
    [
        pygeos.multipoints,
        pygeos.multilinestrings,
        pygeos.multipolygons,
        pygeos.geometrycollections,
    ],
)
@pytest.mark.parametrize(
    "geometries", [np.array([1, 2], dtype=np.intp), None, np.array([[point]]), "hello"]
)
def test_invalid_geometries(func, geometries):
    with pytest.raises((TypeError, ValueError)):
        func(geometries, indices=[0, 1])


@pytest.mark.parametrize(
    "func", [pygeos.points, pygeos.linestrings, pygeos.linearrings]
)
@pytest.mark.parametrize("indices", [[point], " hello", [0, 1], [-1]])
def test_invalid_indices_simple(func, indices):
    with pytest.raises((TypeError, ValueError)):
        func([[0.2, 0.3]], indices=indices)


def test_points_invalid():
    # attempt to construct a point with 2 coordinates
    with pytest.raises(pygeos.GEOSException):
        pygeos.points([[1, 1], [2, 2]], indices=[0, 0])


def test_points():
    actual = pygeos.points(
        np.array([[2, 3], [2, 3]], dtype=float),
        indices=np.array([0, 1], dtype=np.intp),
    )
    assert_geometries_equal(actual, [point, point])


def test_points_no_index_raises():
    with pytest.raises(ValueError):
        pygeos.points(
            np.array([[2, 3], [2, 3]], dtype=float),
            indices=np.array([0, 2], dtype=np.intp),
        )


@pytest.mark.parametrize(
    "coordinates,indices,expected",
    [
        ([[1, 1], [2, 2]], [0, 0], [lstrs([[1, 1], [2, 2]])]),
        ([[1, 1, 1], [2, 2, 2]], [0, 0], [lstrs([[1, 1, 1], [2, 2, 2]])]),
        (
            [[1, 1], [2, 2], [2, 2], [3, 3]],
            [0, 0, 1, 1],
            [lstrs([[1, 1], [2, 2]]), lstrs([[2, 2], [3, 3]])],
        ),
    ],
)
def test_linestrings(coordinates, indices, expected):
    actual = pygeos.linestrings(
        np.array(coordinates, dtype=float), indices=np.array(indices, dtype=np.intp)
    )
    assert_geometries_equal(actual, expected)


def test_linestrings_invalid():
    # attempt to construct linestrings with 1 coordinate
    with pytest.raises(pygeos.GEOSException):
        pygeos.linestrings([[1, 1], [2, 2]], indices=[0, 1])


@pytest.mark.parametrize(
    "coordinates", [([[1, 1], [2, 1], [2, 2], [1, 1]]), ([[1, 1], [2, 1], [2, 2]])]
)
def test_linearrings(coordinates):
    actual = pygeos.linearrings(
        np.array(coordinates, dtype=np.float64),
        indices=np.zeros(len(coordinates), dtype=np.intp),
    )
    assert_geometries_equal(actual, pygeos.linearrings(coordinates))


@pytest.mark.parametrize(
    "coordinates",
    [
        ([[1, 1], [2, 1], [1, 1]]),  # too few coordinates
        ([[1, np.nan], [2, 1], [2, 2], [1, 1]]),  # starting with nan
    ],
)
def test_linearrings_invalid(coordinates):
    # attempt to construct linestrings with 1 coordinate
    with pytest.raises(pygeos.GEOSException):
        pygeos.linearrings(coordinates, indices=np.zeros(len(coordinates)))


hole_1 = pygeos.linearrings([(0.2, 0.2), (0.2, 0.4), (0.4, 0.4)])
hole_2 = pygeos.linearrings([(0.6, 0.6), (0.6, 0.8), (0.8, 0.8)])
poly = pygeos.polygons(linear_ring)
poly_empty = pygeos.Geometry("POLYGON EMPTY")
poly_hole_1 = pygeos.polygons(linear_ring, holes=[hole_1])
poly_hole_2 = pygeos.polygons(linear_ring, holes=[hole_2])
poly_hole_1_2 = pygeos.polygons(linear_ring, holes=[hole_1, hole_2])


@pytest.mark.parametrize(
    "rings,indices,expected",
    [
        ([linear_ring, linear_ring], [0, 1], [poly, poly]),
        ([None, linear_ring], [0, 1], [poly_empty, poly]),
        ([None, linear_ring, None, None], [0, 0, 1, 1], [poly, poly_empty]),
        ([linear_ring, hole_1, linear_ring], [0, 0, 1], [poly_hole_1, poly]),
        ([linear_ring, linear_ring, hole_1], [0, 1, 1], [poly, poly_hole_1]),
        ([None, linear_ring, linear_ring, hole_1], [0, 0, 1, 1], [poly, poly_hole_1]),
        ([linear_ring, None, linear_ring, hole_1], [0, 0, 1, 1], [poly, poly_hole_1]),
        ([linear_ring, None, linear_ring, hole_1], [0, 1, 1, 1], [poly, poly_hole_1]),
        ([linear_ring, linear_ring, None, hole_1], [0, 1, 1, 1], [poly, poly_hole_1]),
        ([linear_ring, linear_ring, hole_1, None], [0, 1, 1, 1], [poly, poly_hole_1]),
        (
            [linear_ring, hole_1, hole_2, linear_ring],
            [0, 0, 0, 1],
            [poly_hole_1_2, poly],
        ),
        (
            [linear_ring, hole_1, linear_ring, hole_2],
            [0, 0, 1, 1],
            [poly_hole_1, poly_hole_2],
        ),
        (
            [linear_ring, linear_ring, hole_1, hole_2],
            [0, 1, 1, 1],
            [poly, poly_hole_1_2],
        ),
        (
            [linear_ring, hole_1, None, hole_2, linear_ring],
            [0, 0, 0, 0, 1],
            [poly_hole_1_2, poly],
        ),
        (
            [linear_ring, hole_1, None, linear_ring, hole_2],
            [0, 0, 0, 1, 1],
            [poly_hole_1, poly_hole_2],
        ),
        (
            [linear_ring, hole_1, linear_ring, None, hole_2],
            [0, 0, 1, 1, 1],
            [poly_hole_1, poly_hole_2],
        ),
    ],
)
def test_polygons(rings, indices, expected):
    actual = pygeos.polygons(
        np.array(rings, dtype=object), indices=np.array(indices, dtype=np.intp)
    )
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize(
    "func",
    [
        pygeos.polygons,
        pygeos.multipoints,
        pygeos.multilinestrings,
        pygeos.multipolygons,
        pygeos.geometrycollections,
    ],
)
@pytest.mark.parametrize("indices", [np.array([point]), " hello", [0, 1], [-1]])
def test_invalid_indices_collections(func, indices):
    with pytest.raises((TypeError, ValueError)):
        func([point], indices=indices)


@pytest.mark.parametrize(
    "geometries,indices,expected",
    [
        ([point, line_string], [0, 0], [geom_coll([point, line_string])]),
        ([point, line_string], [0, 1], [geom_coll([point]), geom_coll([line_string])]),
        ([point, None], [0, 0], [geom_coll([point])]),
        ([point, None], [0, 1], [geom_coll([point]), geom_coll([])]),
        ([None, point, None, None], [0, 0, 1, 1], [geom_coll([point]), geom_coll([])]),
        ([point, None, line_string], [0, 0, 0], [geom_coll([point, line_string])]),
    ],
)
def test_geometrycollections(geometries, indices, expected):
    actual = pygeos.geometrycollections(
        np.array(geometries, dtype=object), indices=indices
    )
    assert_geometries_equal(actual, expected)


def test_geometrycollections_no_index_raises():
    with pytest.raises(ValueError):
        pygeos.geometrycollections(
            np.array([point, line_string], dtype=object), indices=[0, 2]
        )


def test_multipoints():
    actual = pygeos.multipoints(
        np.array([point], dtype=object), indices=np.zeros(1, dtype=np.intp)
    )
    assert_geometries_equal(actual, pygeos.multipoints([point]))


def test_multilinestrings():
    actual = pygeos.multilinestrings(
        np.array([line_string], dtype=object), indices=np.zeros(1, dtype=np.intp)
    )
    assert_geometries_equal(actual, pygeos.multilinestrings([line_string]))


def test_multilinearrings():
    actual = pygeos.multilinestrings(
        np.array([linear_ring], dtype=object), indices=np.zeros(1, dtype=np.intp)
    )
    assert_geometries_equal(actual, pygeos.multilinestrings([linear_ring]))


def test_multipolygons():
    actual = pygeos.multipolygons(
        np.array([polygon], dtype=object), indices=np.zeros(1, dtype=np.intp)
    )
    assert_geometries_equal(actual, pygeos.multipolygons([polygon]))


@pytest.mark.parametrize(
    "geometries,func",
    [
        ([point], pygeos.polygons),
        ([line_string], pygeos.polygons),
        ([polygon], pygeos.polygons),
        ([line_string], pygeos.multipoints),
        ([polygon], pygeos.multipoints),
        ([point], pygeos.multilinestrings),
        ([polygon], pygeos.multilinestrings),
        ([point], pygeos.multipolygons),
        ([line_string], pygeos.multipolygons),
    ],
)
def test_incompatible_types(geometries, func):
    with pytest.raises(TypeError):
        func(geometries, indices=[0])
