import pygeos
import pytest
import numpy as np

from .common import (
    point,
    line_string,
    linear_ring,
    polygon,
    multi_point,
    multi_line_string,
    multi_polygon,
    geometry_collection,
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


@pytest.mark.parametrize("shape", [
    (2, 1, 2),  # 2 linestrings of 1 2D point
    (1, 1, 2),  # 1 linestring of 1 2D point
    (1, 2),  # 1 linestring of 1 2D point (scalar)
    (2, ),  # 1 2D point (scalar)
])
def test_linestrings_invalid_shape(shape):
    with pytest.raises(ValueError):
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


@pytest.mark.parametrize("shape", [
    (2, 1, 2),  # 2 linearrings of 1 2D point
    (1, 1, 2),  # 1 linearring of 1 2D point
    (1, 2),  # 1 linearring of 1 2D point (scalar)
    (2, 2, 2),  # 2 linearrings of 2 2D points
    (1, 2, 2),  # 1 linearring of 2 2D points
    (2, 2),  # 1 linearring of 2 2D points (scalar)
    (2, 3, 2),  # 2 linearrings of 3 2D points
    (1, 3, 2),  # 1 linearring of 3 2D points
    (3, 2),  # 1 linearring of 3 2D points (scalar)
    (2, ),  # 1 2D point (scalar)
])
def test_linearrings_invalid_shape(shape):
    coords = np.ones(shape)
    with pytest.raises(ValueError):
        pygeos.linearrings(coords)

    # make sure the first coordinate != second coordinate
    coords[..., 1] += 1
    with pytest.raises(ValueError):
        pygeos.linearrings(coords)

def test_polygon_from_linearring():
    actual = pygeos.polygons(pygeos.linearrings(box_tpl(0, 0, 1, 1)))
    assert str(actual) == "POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"


def test_polygons():
    actual = pygeos.polygons(box_tpl(0, 0, 1, 1))
    assert str(actual) == "POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"


def test_polygon_no_hole_list_raises():
    with pytest.raises(ValueError):
        pygeos.polygons(box_tpl(0, 0, 10, 10), box_tpl(1, 1, 2, 2))


def test_polygon_with_1_hole():
    actual = pygeos.polygons(box_tpl(0, 0, 10, 10), [box_tpl(1, 1, 2, 2)])
    assert pygeos.area(actual) == 99.0


def test_polygon_with_2_holes():
    actual = pygeos.polygons(
        box_tpl(0, 0, 10, 10), [box_tpl(1, 1, 2, 2), box_tpl(3, 3, 4, 4)]
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


@pytest.mark.parametrize("shape", [
    (2, 1, 2),  # 2 linearrings of 1 2D point
    (1, 1, 2),  # 1 linearring of 1 2D point
    (1, 2),  # 1 linearring of 1 2D point (scalar)
    (2, 2, 2),  # 2 linearrings of 2 2D points
    (1, 2, 2),  # 1 linearring of 2 2D points
    (2, 2),  # 1 linearring of 2 2D points (scalar)
    (2, 3, 2),  # 2 linearrings of 3 2D points
    (1, 3, 2),  # 1 linearring of 3 2D points
    (3, 2),  # 1 linearring of 3 2D points (scalar)
    (2, ),  # 1 2D point (scalar)
])
def test_polygons_not_enough_points_in_shell(shape):
    coords = np.ones(shape)
    with pytest.raises(ValueError):
        pygeos.polygons(coords)
    
    # make sure the first coordinate != second coordinate
    coords[..., 1] += 1
    with pytest.raises(ValueError):
        pygeos.polygons(coords)


@pytest.mark.parametrize("shape", [
    (2, 1, 2),  # 2 linearrings of 1 2D point
    (1, 1, 2),  # 1 linearring of 1 2D point
    (1, 2),  # 1 linearring of 1 2D point (scalar)
    (2, 2, 2),  # 2 linearrings of 2 2D points
    (1, 2, 2),  # 1 linearring of 2 2D points
    (2, 2),  # 1 linearring of 2 2D points (scalar)
    (2, 3, 2),  # 2 linearrings of 3 2D points
    (1, 3, 2),  # 1 linearring of 3 2D points
    (3, 2),  # 1 linearring of 3 2D points (scalar)
    (2, ),  # 1 2D point (scalar)
])
def test_polygons_not_enough_points_in_holes(shape):
    coords = np.ones(shape)
    with pytest.raises(ValueError):
        pygeos.polygons(np.ones((1, 4, 2)), coords)
    
    # make sure the first coordinate != second coordinate
    coords[..., 1] += 1
    with pytest.raises(ValueError):
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


def test_box():
    actual = pygeos.box(0, 0, 1, 1)
    assert str(actual) == "POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"


def test_box_multiple():
    actual = pygeos.box(0, 0, [1, 2], [1, 2])
    assert str(actual[0]) == "POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"
    assert str(actual[1]) == "POLYGON ((2 0, 2 2, 0 2, 0 0, 2 0))"


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
    for point in [Point("POINT (1 1)"), pygeos.points(1, 1)]:
        assert isinstance(point, Point)
        assert pygeos.get_type_id(point) == pygeos.GeometryType.POINT
        assert point.x == 1


def test_subclass_is_geometry(with_point_in_registry):
    assert pygeos.is_geometry(Point("POINT (1 1)"))


def test_subclass_is_valid_input(with_point_in_registry):
    assert pygeos.is_valid_input(Point("POINT (1 1)"))
