import numpy as np
import pygeos
from pygeos import Point, LineString, box

point_polygon_testdata = [Point(i, i) for i in range(6)], box(2, 2, 4, 4)


# Y_b

def test_has_z():
    geoms = [Point(1.0, 1.0), Point(1.0, 1.0, 1.0)]
    actual = pygeos.has_z(geoms)
    expected = [False, True]
    np.testing.assert_equal(actual, expected)

# YY_b

def test_disjoint():
    points, polygon = point_polygon_testdata
    actual = pygeos.disjoint(polygon, points)
    expected = [True, True, False, False, False, True]
    np.testing.assert_equal(actual, expected)


def test_touches():
    points, polygon = point_polygon_testdata
    actual = pygeos.touches(polygon, points)
    expected = [False, False, True, False, True, False]
    np.testing.assert_equal(actual, expected)


def test_intersects():
    points, polygon = point_polygon_testdata
    actual = pygeos.intersects(polygon, points)
    expected = [False, False, True, True, True, False]
    np.testing.assert_equal(actual, expected)


def test_within():
    points, polygon = point_polygon_testdata
    actual = pygeos.within(points, polygon)
    expected = [False, False, False, True, False, False]
    np.testing.assert_equal(actual, expected)


def test_contains():
    points, polygon = point_polygon_testdata
    actual = pygeos.contains(polygon, points)
    expected = [False, False, False, True, False, False]
    np.testing.assert_equal(actual, expected)

# Y_Y

def test_get_centroid():
    poly = box(0, 0, 10, 10)
    actual = pygeos.get_centroid(poly)
    expected = Point(5, 5)
    assert pygeos.equals(actual, expected)


# YY_Y

def test_intersection():
    poly1, poly2 = box(0, 0, 10, 10), box(5, 5, 20, 20)
    actual = pygeos.intersection(poly1, poly2)
    expected = box(5, 5, 10, 10)
    assert pygeos.equals(actual, expected)


def test_union():
    poly1, poly2 = box(0, 0, 10, 10), box(10, 0, 20, 10)
    actual = pygeos.union(poly1, poly2)
    expected = box(0, 0, 20, 10)
    assert pygeos.equals(actual, expected)

# Y_d


def test_area():
    poly = box(0, 0, 10, 10)
    assert pygeos.area(poly) == 100.

# Y_B


def test_geom_type_id():
    line = LineString([[0, 0], [1, 1], [2, 1]])
    poly = box(0, 0, 10, 10)
    assert pygeos.geom_type_id([line, poly]).tolist() == [1, 3]

# Y_l


def test_get_num_points():
    line = LineString([[0, 0], [1, 1], [2, 1]])
    assert pygeos.get_num_points(line) == 3


# Y_l


def test_distance():
    points, polygon = point_polygon_testdata
    actual = pygeos.distance(points, polygon)
    expected = [2 * 2**0.5, 2**0.5, 0, 0, 0, 2**0.5]
    np.testing.assert_allclose(actual, expected)
