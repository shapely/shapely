import pygeos
import pytest
import numpy as np

from .common import point


def box_tpl(x1, y1, x2, y2):
    return (x2, y1), (x2, y2), (x1, y2), (x1, y1), (x2, y1)


def test_points_from_coords():
    actual = pygeos.points([[0, 0], [2, 2]])
    assert str(actual[0]) == "POINT (0 0)"
    assert str(actual[1])  == "POINT (2 2)"


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


def test_linearrings():
    actual = pygeos.linearrings(box_tpl(0, 0, 1, 1))
    assert str(actual) == "LINEARRING (1 0, 1 1, 0 1, 0 0, 1 0)"


def test_linearrings_from_xy():
    actual = pygeos.linearrings([0, 1, 2, 0], [3, 4, 5, 3])
    assert str(actual) == "LINEARRING (0 3, 1 4, 2 5, 0 3)"


def test_linearrings_unclosed():
    actual = pygeos.linearrings(box_tpl(0, 0, 1, 1)[:-1])
    assert str(actual) == "LINEARRING (1 0, 1 1, 0 1, 0 0, 1 0)"


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


def test_create_collection_only_none():
    actual = pygeos.multipoints(np.array([None], dtype=object))
    assert str(actual) == "MULTIPOINT EMPTY"


def test_create_collection_skips_none():
    actual = pygeos.multipoints([point, None, None, point])
    assert str(actual) == "MULTIPOINT (2 3, 2 3)"


def test_create_collection_wrong_collection_type():
    with pytest.raises(TypeError):
        pygeos.lib.create_collection([point], 2)


def test_create_collection_wrong_geom_type():
    with pytest.raises(TypeError):
        pygeos.multipolygons([point])


def test_box():
    actual = pygeos.box(0, 0, 1, 1)
    assert str(actual) == "POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"


def test_box_multiple():
    actual = pygeos.box(0, 0, [1, 2], [1, 2])
    assert str(actual[0]) == "POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))"
    assert str(actual[1]) == "POLYGON ((2 0, 2 2, 0 2, 0 0, 2 0))"
