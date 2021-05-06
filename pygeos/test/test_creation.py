import numpy as np
import pytest

import pygeos

from .common import (
    empty_polygon,
    geometry_collection,
    line_string,
    linear_ring,
    multi_line_string,
    multi_point,
    multi_polygon,
    point,
    polygon,
)


def box_tpl(x1, y1, x2, y2):
    return (x2, y1), (x2, y2), (x1, y2), (x1, y1), (x2, y1)


def test_points_from_coords():
    actual = pygeos.points([[0, 0], [2, 2]])
    assert str(actual[0]) == "POINT (0 0)"
    assert str(actual[1]) == "POINT (2 2)"


def test_points_from_xy():
    actual = pygeos.points(2, [0, 1])
    assert str(actual[0]) == "POINT (2 0)"
    assert str(actual[1]) == "POINT (2 1)"


def test_points_from_xyz():
    actual = pygeos.points(1, 1, [0, 1])
    assert str(actual[0]) == "POINT Z (1 1 0)"
    assert str(actual[1]) == "POINT Z (1 1 1)"


def test_points_invalid_ndim():
    with pytest.raises(pygeos.GEOSException):
        pygeos.points([0, 1, 2, 3])


@pytest.mark.skipif(pygeos.geos_version < (3, 10, 0), reason="GEOS < 3.10")
def test_points_nan_becomes_empty():
    assert str(pygeos.points(np.nan, np.nan)) == "POINT EMPTY"


def test_linestrings_from_coords():
    actual = pygeos.linestrings([[[0, 0], [1, 1]], [[0, 0], [2, 2]]])
    assert str(actual[0]) == "LINESTRING (0 0, 1 1)"
    assert str(actual[1]) == "LINESTRING (0 0, 2 2)"


def test_linestrings_from_xy():
    actual = pygeos.linestrings([0, 1], [2, 3])
    assert str(actual) == "LINESTRING (0 2, 1 3)"


def test_linestrings_from_xy_broadcast():
    x = [0, 1]  # the same X coordinates for both linestrings
    y = [2, 3], [4, 5]  # each linestring has a different set of Y coordinates
    actual = pygeos.linestrings(x, y)
    assert str(actual[0]) == "LINESTRING (0 2, 1 3)"
    assert str(actual[1]) == "LINESTRING (0 4, 1 5)"


def test_linestrings_from_xyz():
    actual = pygeos.linestrings([0, 1], [2, 3], 0)
    assert str(actual) == "LINESTRING Z (0 2 0, 1 3 0)"


def test_linestrings_invalid_shape_scalar():
    with pytest.raises(ValueError):
        pygeos.linestrings((1, 1))


@pytest.mark.parametrize(
    "shape",
    [
        (2, 1, 2),  # 2 linestrings of 1 2D point
        (1, 1, 2),  # 1 linestring of 1 2D point
        (1, 2),  # 1 linestring of 1 2D point (scalar)
    ],
)
def test_linestrings_invalid_shape(shape):
    with pytest.raises(pygeos.GEOSException):
        pygeos.linestrings(np.ones(shape))


def test_linearrings():
    actual = pygeos.linearrings(box_tpl(0, 0, 1, 1))
    assert str(actual) == "LINEARRING (1 0, 1 1, 0 1, 0 0, 1 0)"


def test_linearrings_from_xy():
    actual = pygeos.linearrings([0, 1, 2, 0], [3, 4, 5, 3])
    assert str(actual) == "LINEARRING (0 3, 1 4, 2 5, 0 3)"


def test_linearrings_unclosed():
    actual = pygeos.linearrings(box_tpl(0, 0, 1, 1)[:-1])
    assert str(actual) == "LINEARRING (1 0, 1 1, 0 1, 0 0, 1 0)"


def test_linearrings_invalid_shape_scalar():
    with pytest.raises(ValueError):
        pygeos.linearrings((1, 1))


@pytest.mark.parametrize(
    "shape",
    [
        (2, 1, 2),  # 2 linearrings of 1 2D point
        (1, 1, 2),  # 1 linearring of 1 2D point
        (1, 2),  # 1 linearring of 1 2D point (scalar)
        (2, 2, 2),  # 2 linearrings of 2 2D points
        (1, 2, 2),  # 1 linearring of 2 2D points
        (2, 2),  # 1 linearring of 2 2D points (scalar)
        (2, 3, 2),  # 2 linearrings of 3 2D points
        (1, 3, 2),  # 1 linearring of 3 2D points
        (3, 2),  # 1 linearring of 3 2D points (scalar)
    ],
)
def test_linearrings_invalid_shape(shape):
    coords = np.ones(shape)
    with pytest.raises(pygeos.GEOSException):
        pygeos.linearrings(coords)

    # make sure the first coordinate != second coordinate
    coords[..., 1] += 1
    with pytest.raises(pygeos.GEOSException):
        pygeos.linearrings(coords)


def test_linearrings_all_nan():
    coords = np.full((4, 2), np.nan)
    with pytest.raises(pygeos.GEOSException):
        pygeos.linearrings(coords)


def test_polygon_from_linearring():
    actual = pygeos.polygons(pygeos.linearrings(box_tpl(0, 0, 1, 1)))
    assert str(actual) == "POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"


def test_polygons_none():
    assert pygeos.equals(pygeos.polygons(None), empty_polygon)
    assert pygeos.equals(pygeos.polygons(None, holes=[linear_ring]), empty_polygon)


def test_polygons():
    actual = pygeos.polygons(box_tpl(0, 0, 1, 1))
    assert str(actual) == "POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"


def test_polygon_no_hole_list_raises():
    with pytest.raises(ValueError):
        pygeos.polygons(box_tpl(0, 0, 10, 10), box_tpl(1, 1, 2, 2))


def test_polygon_no_hole_wrong_type():
    with pytest.raises((TypeError, pygeos.GEOSException)):
        pygeos.polygons(point)


def test_polygon_with_hole_wrong_type():
    with pytest.raises((TypeError, pygeos.GEOSException)):
        pygeos.polygons(point, [linear_ring])


def test_polygon_wrong_hole_type():
    with pytest.raises((TypeError, pygeos.GEOSException)):
        pygeos.polygons(linear_ring, [point])


def test_polygon_with_1_hole():
    actual = pygeos.polygons(box_tpl(0, 0, 10, 10), [box_tpl(1, 1, 2, 2)])
    assert pygeos.area(actual) == 99.0


def test_polygon_with_2_holes():
    actual = pygeos.polygons(
        box_tpl(0, 0, 10, 10), [box_tpl(1, 1, 2, 2), box_tpl(3, 3, 4, 4)]
    )
    assert pygeos.area(actual) == 98.0


def test_polygon_with_none_hole():
    actual = pygeos.polygons(
        pygeos.linearrings(box_tpl(0, 0, 10, 10)),
        [
            pygeos.linearrings(box_tpl(1, 1, 2, 2)),
            None,
            pygeos.linearrings(box_tpl(3, 3, 4, 4)),
        ],
    )
    assert pygeos.area(actual) == 98.0


def test_2_polygons_with_same_hole():
    actual = pygeos.polygons(
        [box_tpl(0, 0, 10, 10), box_tpl(0, 0, 5, 5)], [box_tpl(1, 1, 2, 2)]
    )
    assert pygeos.area(actual).tolist() == [99.0, 24.0]


def test_2_polygons_with_2_same_holes():
    actual = pygeos.polygons(
        [box_tpl(0, 0, 10, 10), box_tpl(0, 0, 5, 5)],
        [box_tpl(1, 1, 2, 2), box_tpl(3, 3, 4, 4)],
    )
    assert pygeos.area(actual).tolist() == [98.0, 23.0]


def test_2_polygons_with_different_holes():
    actual = pygeos.polygons(
        [box_tpl(0, 0, 10, 10), box_tpl(0, 0, 5, 5)],
        [[box_tpl(1, 1, 3, 3)], [box_tpl(1, 1, 2, 2)]],
    )
    assert pygeos.area(actual).tolist() == [96.0, 24.0]


def test_polygons_not_enough_points_in_shell_scalar():
    with pytest.raises(ValueError):
        pygeos.polygons((1, 1))


@pytest.mark.parametrize(
    "shape",
    [
        (2, 1, 2),  # 2 linearrings of 1 2D point
        (1, 1, 2),  # 1 linearring of 1 2D point
        (1, 2),  # 1 linearring of 1 2D point (scalar)
        (2, 2, 2),  # 2 linearrings of 2 2D points
        (1, 2, 2),  # 1 linearring of 2 2D points
        (2, 2),  # 1 linearring of 2 2D points (scalar)
        (2, 3, 2),  # 2 linearrings of 3 2D points
        (1, 3, 2),  # 1 linearring of 3 2D points
        (3, 2),  # 1 linearring of 3 2D points (scalar)
    ],
)
def test_polygons_not_enough_points_in_shell(shape):
    coords = np.ones(shape)
    with pytest.raises(pygeos.GEOSException):
        pygeos.polygons(coords)

    # make sure the first coordinate != second coordinate
    coords[..., 1] += 1
    with pytest.raises(pygeos.GEOSException):
        pygeos.polygons(coords)


def test_polygons_not_enough_points_in_holes_scalar():
    with pytest.raises(ValueError):
        pygeos.polygons(np.ones((1, 4, 2)), (1, 1))


@pytest.mark.parametrize(
    "shape",
    [
        (2, 1, 2),  # 2 linearrings of 1 2D point
        (1, 1, 2),  # 1 linearring of 1 2D point
        (1, 2),  # 1 linearring of 1 2D point (scalar)
        (2, 2, 2),  # 2 linearrings of 2 2D points
        (1, 2, 2),  # 1 linearring of 2 2D points
        (2, 2),  # 1 linearring of 2 2D points (scalar)
        (2, 3, 2),  # 2 linearrings of 3 2D points
        (1, 3, 2),  # 1 linearring of 3 2D points
        (3, 2),  # 1 linearring of 3 2D points (scalar)
    ],
)
def test_polygons_not_enough_points_in_holes(shape):
    coords = np.ones(shape)
    with pytest.raises(pygeos.GEOSException):
        pygeos.polygons(np.ones((1, 4, 2)), coords)

    # make sure the first coordinate != second coordinate
    coords[..., 1] += 1
    with pytest.raises(pygeos.GEOSException):
        pygeos.polygons(np.ones((1, 4, 2)), coords)


@pytest.mark.parametrize(
    "func,expected",
    [
        (pygeos.multipoints, "MULTIPOINT EMPTY"),
        (pygeos.multilinestrings, "MULTILINESTRING EMPTY"),
        (pygeos.multipolygons, "MULTIPOLYGON EMPTY"),
        (pygeos.geometrycollections, "GEOMETRYCOLLECTION EMPTY"),
    ],
)
def test_create_collection_only_none(func, expected):
    actual = func(np.array([None], dtype=object))
    assert str(actual) == expected


@pytest.mark.parametrize(
    "func,sub_geom",
    [
        (pygeos.multipoints, point),
        (pygeos.multilinestrings, line_string),
        (pygeos.multilinestrings, linear_ring),
        (pygeos.multipolygons, polygon),
        (pygeos.geometrycollections, point),
        (pygeos.geometrycollections, line_string),
        (pygeos.geometrycollections, linear_ring),
        (pygeos.geometrycollections, polygon),
        (pygeos.geometrycollections, multi_point),
        (pygeos.geometrycollections, multi_line_string),
        (pygeos.geometrycollections, multi_polygon),
        (pygeos.geometrycollections, geometry_collection),
    ],
)
def test_create_collection(func, sub_geom):
    actual = func([sub_geom, sub_geom])
    assert pygeos.get_num_geometries(actual) == 2


@pytest.mark.parametrize(
    "func,sub_geom",
    [
        (pygeos.multipoints, point),
        (pygeos.multilinestrings, line_string),
        (pygeos.multipolygons, polygon),
        (pygeos.geometrycollections, polygon),
    ],
)
def test_create_collection_skips_none(func, sub_geom):
    actual = func([sub_geom, None, None, sub_geom])
    assert pygeos.get_num_geometries(actual) == 2


@pytest.mark.parametrize(
    "func,sub_geom",
    [
        (pygeos.multipoints, line_string),
        (pygeos.multipoints, geometry_collection),
        (pygeos.multipoints, multi_point),
        (pygeos.multilinestrings, point),
        (pygeos.multilinestrings, polygon),
        (pygeos.multilinestrings, multi_line_string),
        (pygeos.multipolygons, linear_ring),
        (pygeos.multipolygons, multi_point),
        (pygeos.multipolygons, multi_polygon),
    ],
)
def test_create_collection_wrong_geom_type(func, sub_geom):
    with pytest.raises(TypeError):
        func([sub_geom])


@pytest.mark.parametrize(
    "coords,ccw,expected",
    [
        ((0, 0, 1, 1), True, pygeos.Geometry("POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))")),
        ((0, 0, 1, 1), False, pygeos.Geometry("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")),
    ],
)
def test_box(coords, ccw, expected):
    actual = pygeos.box(*coords, ccw=ccw)
    assert pygeos.equals(actual, expected)


@pytest.mark.parametrize(
    "coords,ccw,expected",
    [
        (
            (0, 0, [1, 2], [1, 2]),
            True,
            [
                pygeos.Geometry("POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"),
                pygeos.Geometry("POLYGON ((2 0, 2 2, 0 2, 0 0, 2 0))"),
            ],
        ),
        (
            (0, 0, [1, 2], [1, 2]),
            [True, False],
            [
                pygeos.Geometry("POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"),
                pygeos.Geometry("POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))"),
            ],
        ),
    ],
)
def test_box_array(coords, ccw, expected):
    actual = pygeos.box(*coords, ccw=ccw)
    assert pygeos.equals(actual, expected).all()


@pytest.mark.parametrize(
    "coords",
    [
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, 0, 1, 1],
        [0, np.nan, 1, 1],
        [0, 0, np.nan, 1],
        [0, 0, 1, np.nan],
    ],
)
def test_box_nan(coords):
    assert pygeos.box(*coords) is None


class BaseGeometry(pygeos.Geometry):
    @property
    def type_id(self):
        return pygeos.get_type_id(self)


class Point(BaseGeometry):
    @property
    def x(self):
        return pygeos.get_x(self)

    @property
    def y(self):
        return pygeos.get_y(self)


@pytest.fixture
def with_point_in_registry():
    orig = pygeos.lib.registry[0]
    pygeos.lib.registry[0] = Point
    yield
    pygeos.lib.registry[0] = orig


def test_subclasses(with_point_in_registry):
    for _point in [Point("POINT (1 1)"), pygeos.points(1, 1)]:
        assert isinstance(_point, Point)
        assert pygeos.get_type_id(_point) == pygeos.GeometryType.POINT
        assert _point.x == 1


def test_prepare():
    arr = np.array([pygeos.points(1, 1), None, pygeos.box(0, 0, 1, 1)])
    assert arr[0]._ptr_prepared == 0
    assert arr[2]._ptr_prepared == 0
    pygeos.prepare(arr)
    assert arr[0]._ptr_prepared != 0
    assert arr[1] is None
    assert arr[2]._ptr_prepared != 0

    # preparing again actually does nothing
    original = arr[0]._ptr_prepared
    pygeos.prepare(arr)
    assert arr[0]._ptr_prepared == original


def test_destroy_prepared():
    arr = np.array([pygeos.points(1, 1), None, pygeos.box(0, 0, 1, 1)])
    pygeos.prepare(arr)
    assert arr[0]._ptr_prepared != 0
    assert arr[2]._ptr_prepared != 0
    pygeos.destroy_prepared(arr)
    assert arr[0]._ptr_prepared == 0
    assert arr[1] is None
    assert arr[2]._ptr_prepared == 0
    pygeos.destroy_prepared(arr)  # does not error


def test_subclass_is_geometry(with_point_in_registry):
    assert pygeos.is_geometry(Point("POINT (1 1)"))


def test_subclass_is_valid_input(with_point_in_registry):
    assert pygeos.is_valid_input(Point("POINT (1 1)"))
