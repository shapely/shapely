import pytest
import pygeos
from pygeos import apply, count_coordinates, get_coordinates, set_coordinates
import numpy as np
from numpy.testing import assert_equal, assert_allclose

from .common import empty
from .common import point
from .common import empty_point
from .common import point_z
from .common import line_string
from .common import line_string_z
from .common import linear_ring
from .common import polygon
from .common import polygon_z
from .common import polygon_with_hole
from .common import multi_point
from .common import multi_line_string
from .common import multi_polygon
from .common import geometry_collection
from .common import geometry_collection_z

nested_2 = pygeos.geometrycollections([geometry_collection, point])
nested_3 = pygeos.geometrycollections([nested_2, point])


@pytest.mark.parametrize(
    "geoms,count",
    [
        ([], 0),
        ([empty], 0),
        ([point, empty], 1),
        ([empty, point, empty], 1),
        ([point, None], 1),
        ([None, point, None], 1),
        ([point, point], 2),
        ([point, point_z], 2),
        ([line_string, linear_ring], 8),
        ([polygon], 5),
        ([polygon_with_hole], 10),
        ([multi_point, multi_line_string], 4),
        ([multi_polygon], 10),
        ([geometry_collection], 3),
        ([nested_2], 4),
        ([nested_3], 5),
    ],
)
def test_count_coords(geoms, count):
    actual = count_coordinates(np.array(geoms, np.object))
    assert actual == count


# fmt: off
@pytest.mark.parametrize("include_z", [True, False])
@pytest.mark.parametrize(
    "geoms,x,y",
    [
        ([], [], []),
        ([empty], [], []),
        ([point, empty], [2], [3]),
        ([empty, point, empty], [2], [3]),
        ([point, None], [2], [3]),
        ([None, point, None], [2], [3]),
        ([point, point], [2, 2], [3, 3]),
        ([line_string, linear_ring], [0, 1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 0]),
        ([polygon], [0, 2, 2, 0, 0], [0, 0, 2, 2, 0]),
        ([polygon_with_hole], [0, 0, 10, 10, 0, 2, 2, 4, 4, 2], [0, 10, 10, 0, 0, 2, 4, 4, 2, 2]),
        ([multi_point, multi_line_string], [0, 1, 0, 1], [0, 2, 0, 2]),
        ([multi_polygon], [0, 1, 1, 0, 0, 2.1, 2.2, 2.2, 2.1, 2.1], [0, 0, 1, 1, 0, 2.1, 2.1, 2.2, 2.2, 2.1]),
        ([geometry_collection], [51, 52, 49], [-1, -1, 2]),
        ([nested_2], [51, 52, 49, 2], [-1, -1, 2, 3]),
        ([nested_3], [51, 52, 49, 2, 2], [-1, -1, 2, 3, 3]),
    ],
)  # fmt: on
def test_get_coords(geoms, x, y, include_z):
    actual = get_coordinates(np.array(geoms, np.object), include_z=include_z)
    if not include_z:
        expected = np.array([x, y], np.float64).T
    else:
        expected = np.array([x, y, [np.nan]*len(x)], np.float64).T
    assert_equal(actual, expected)


# fmt: off
@pytest.mark.parametrize("include_z", [True, False])
@pytest.mark.parametrize(
    "geoms,x,y,z",
    [
        ([point, point_z], [2, 1], [3, 1], [np.nan, 1]),
        ([line_string_z], [0, 1, 1], [0, 0, 1], [0, 1, 2]),
        ([polygon_z], [0, 2, 2, 0, 0], [0, 0, 2, 2, 0], [0, 1, 2, 3, 0]),
        ([geometry_collection_z], [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 1, 2]),
    ],
)  # fmt: on
def test_get_coords_3d(geoms, x, y, z, include_z):
    actual = get_coordinates(np.array(geoms, np.object), include_z=include_z)
    if include_z:
        expected = np.array([x, y, z], np.float64).T
    else:
        expected = np.array([x, y], np.float64).T
    assert_equal(actual, expected)


@pytest.mark.parametrize("include_z", [True, False])
@pytest.mark.parametrize(
    "geoms,count,has_ring",
    [
        ([], 0, False),
        ([empty], 0, False),
        ([empty_point], 0, False),
        ([point, empty], 1, False),
        ([empty, point, empty], 1, False),
        ([point, None], 1, False),
        ([None, point, None], 1, False),
        ([point, point], 2, False),
        ([point, point_z], 2, False),
        ([line_string, linear_ring], 8, True),
        ([line_string_z], 3, True),
        ([polygon], 5, True),
        ([polygon_z], 5, True),
        ([polygon_with_hole], 10, True),
        ([multi_point, multi_line_string], 4, False),
        ([multi_polygon], 10, True),
        ([geometry_collection], 3, False),
        ([geometry_collection_z], 3, False),
        ([nested_2], 4, False),
        ([nested_3], 5, False),
    ],
)
def test_set_coords(geoms, count, has_ring, include_z):
    arr_geoms = np.array(geoms, np.object)
    n = 3 if include_z else 2
    coords = get_coordinates(arr_geoms, include_z=include_z) + np.random.random((1, n))
    new_geoms = set_coordinates(arr_geoms, coords)
    assert_equal(coords, get_coordinates(new_geoms, include_z=include_z))


def test_set_coords_nan():
    geoms = np.array([point])
    coords = np.array([[np.nan, np.inf]])
    new_geoms = set_coordinates(geoms, coords)
    assert_equal(coords, get_coordinates(new_geoms))


def test_set_coords_breaks_ring():
    with pytest.raises(pygeos.GEOSException):
        set_coordinates(linear_ring, np.random.random((5, 2)))


def test_set_coords_0dim():
    # a geometry input returns a geometry
    actual = set_coordinates(point, [[1, 1]])
    assert isinstance(actual, pygeos.Geometry)
    # a 0-dim array input returns a 0-dim array
    actual = set_coordinates(np.asarray(point), [[1, 1]])
    assert isinstance(actual, np.ndarray)
    assert actual.ndim == 0


@pytest.mark.parametrize("include_z", [True, False])
def test_set_coords_mixed_dimension(include_z):
    geoms = np.array([point, point_z], dtype=object)
    coords = get_coordinates(geoms, include_z=include_z)
    new_geoms = set_coordinates(geoms, coords * 2)
    if include_z:
        # preserve original dimensionality
        assert not pygeos.has_z(new_geoms[0])
        assert pygeos.has_z(new_geoms[1])
    else:
        # all 2D
        assert not pygeos.has_z(new_geoms).any()


@pytest.mark.parametrize("include_z", [True, False])
@pytest.mark.parametrize(
    "geoms",
    [[], [empty], [None, point, None], [nested_3], [point, point_z], [line_string_z]],
)
def test_apply(geoms, include_z):
    geoms = np.array(geoms, np.object)
    coordinates_before = get_coordinates(geoms, include_z=include_z)
    new_geoms = apply(geoms, lambda x: x + 1, include_z=include_z)
    assert new_geoms is not geoms
    coordinates_after = get_coordinates(new_geoms, include_z=include_z)
    assert_allclose(coordinates_before + 1, coordinates_after, equal_nan=True)


def test_apply_0dim():
    # a geometry input returns a geometry
    actual = apply(point, lambda x: x + 1)
    assert isinstance(actual, pygeos.Geometry)
    # a 0-dim array input returns a 0-dim array
    actual = apply(np.asarray(point), lambda x: x + 1)
    assert isinstance(actual, np.ndarray)
    assert actual.ndim == 0


def test_apply_check_shape():
    def remove_coord(arr):
        return arr[:-1]

    with pytest.raises(ValueError):
        apply(linear_ring, remove_coord)


def test_apply_correct_coordinate_dimension():
    # ensure that new geometry is 2D with include_z=False
    geom = line_string_z
    assert pygeos.get_coordinate_dimension(geom) == 3
    new_geom = apply(geom, lambda x: x + 1, include_z=False)
    assert pygeos.get_coordinate_dimension(new_geom) == 2
