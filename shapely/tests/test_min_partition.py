from shapely.algorithms.min_partition import *
from shapely.geometry import Point, Polygon


def test_minimum_edge_length_partition():
    # Test case 1: Rectilinear polygon with minimum edge length partitioning
    polygon_points = [
        Point(1, 5),
        Point(1, 4),
        Point(3, 4),
        Point(3, 3),
        Point(2, 3),
        Point(2, 1),
        Point(5, 1),
        Point(8, 2),
        Point(8, 1),
        Point(9, 1),
        Point(9, 4),
        Point(8, 4),
        Point(8, 5),
    ]
    assert minimum_edge_length_partition(polygon_points) == [
        [(Point(3, 2), Point(3, 3))],
        [(Point(3, 2), Point(5, 2))],
        [(Point(8, 2), Point(8, 4))],
        [(Point(3, 4), Point(3, 5))],
        [(Point(3, 1), Point(3, 2))],
    ]

    # Test case 2: Rectilinear polygon with no minimum edge length partitioning (a square)
    polygon_points = [Point(0, 0), Point(4, 0), Point(4, 4), Point(0, 4)]
    assert minimum_edge_length_partition(polygon_points) == None

    # Test case 3: Rectilinear polygon with multiple minimum edge length partitionings
    polygon_points = [
        Point(0, 0),
        Point(0, 3),
        Point(3, 3),
        Point(4, 3),
        Point(4, 2),
        Point(12, 2),
        Point(12, 10),
        Point(10, 10),
        Point(10, 8),
        Point(6, 8),
        Point(6, 14),
        Point(0, 14),
    ]
    assert minimum_edge_length_partition(polygon_points) == [
        [(Point(0, 3), Point(3, 3))],
        [(Point(4, 3), Point(4, 8))],
        [(Point(0, 8), Point(6, 8))],
        [(Point(10, 8), Point(8, 2))],
        [(Point(10, 8), Point(12, 10))],
    ]

    # Test case 4: Rectilinear polygon with only one minimum edge length partitioning
    polygon_points = [
        Point(2, 0),
        Point(6, 0),
        Point(6, 4),
        Point(8, 4),
        Point(6, 8),
        Point(0, 6),
        Point(0, 4),
        Point(2, 4),
    ]
    assert minimum_edge_length_partition(polygon_points) == [
        [(Point(2, 4), Point(6, 4))]
    ]

    # Test case 5: Rectilinear polygon with no vertices
    polygon_points = []
    assert minimum_edge_length_partition(polygon_points) == None

    # Test case 6: non-Rectilinear polygon
    polygon_points = [Point(0, 0), Point(3, 0), Point(0, 6)]
    assert minimum_edge_length_partition(polygon_points) == None


test_minimum_edge_length_partition()


def test_find_convex_points():
    # Test case 1: Convex polygon
    polygon_points = [Point(0, 0), Point(0, 2), Point(2, 2), Point(2, 0)]
    assert find_convex_points(polygon_points) == [
        Point(0, 0),
        Point(0, 2),
        Point(2, 2),
        Point(2, 0),
    ]

    # Test case 2: Non-convex polygon
    polygon_points = [Point(0, 0), Point(0, 2), Point(1, 1), Point(2, 2), Point(2, 0)]
    assert find_convex_points(polygon_points) == [
        Point(0, 0),
        Point(0, 2),
        Point(2, 2),
        Point(2, 0),
    ]

    # Test case 3: Convex polygon with duplicate points
    polygon_points = [Point(0, 0), Point(0, 2), Point(2, 2), Point(2, 0), Point(0, 0)]
    assert find_convex_points(polygon_points) == [
        Point(0, 0),
        Point(0, 2),
        Point(2, 2),
        Point(2, 0),
    ]

    # Test case 4: Convex polygon with collinear points
    polygon_points = [
        Point(0, 0),
        Point(0, 2),
        Point(0, 4),
        Point(2, 4),
        Point(2, 2),
        Point(2, 0),
    ]
    assert find_convex_points(polygon_points) == [
        Point(0, 0),
        Point(0, 4),
        Point(2, 4),
        Point(2, 0),
    ]

    # Test case 5: Empty polygon
    polygon_points = []
    assert find_convex_points(polygon_points) == []


test_find_convex_points()


def test_find_matching_point():
    # Test case 1: Matching point exists
    polygon_points = [
        Point(2, 0),
        Point(6, 0),
        Point(6, 4),
        Point(8, 4),
        Point(6, 8),
        Point(0, 6),
        Point(0, 4),
        Point(2, 4),
    ]
    assert find_matching_point(Point(6, 0), polygon_points) == [
        Point(2, 4),
        Point(2, 6),
    ]

    # Test case 2: Matching point does not exist
    polygon_points = [
        Point(1, 5),
        Point(1, 4),
        Point(3, 4),
        Point(3, 3),
        Point(2, 3),
        Point(2, 1),
        Point(5, 1),
        Point(8, 2),
        Point(8, 1),
        Point(9, 1),
        Point(9, 4),
        Point(8, 4),
        Point(8, 5),
    ]
    assert find_matching_point(Point(8, 2), polygon_points) == None

    # Test case 3: Multiple matching points exist
    polygon_points = [
        Point(1, 5),
        Point(1, 4),
        Point(3, 4),
        Point(3, 3),
        Point(2, 3),
        Point(2, 1),
        Point(5, 1),
        Point(8, 2),
        Point(8, 1),
        Point(9, 1),
        Point(9, 4),
        Point(8, 4),
        Point(8, 5),
    ]
    assert find_matching_point(Point(6, 1), polygon_points) == [
        Point(2, 1),
        Point(3, 1),
    ]

    # Test case 4: Empty polygon
    polygon_points = []
    assert find_matching_point(Point(2, 0), polygon_points) == None

    # Test case 5: Matching point on the boundary
    polygon_points = [
        Point(0, 0),
        Point(0, 3),
        Point(3, 3),
        Point(4, 3),
        Point(4, 2),
        Point(12, 2),
        Point(12, 10),
        Point(10, 10),
        Point(10, 8),
        Point(6, 8),
        Point(6, 14),
        Point(0, 14),
    ]
    assert find_matching_point(Point(8, 2), polygon_points) == [
        Point(3, 5),
        Point(3, 4),
        Point(3, 3),
    ]


test_find_matching_point()


def test_find_blocked_rectangle():
    # Test case 1: Blocked rectangle exists
    polygon_points = [
        Point(2, 0),
        Point(6, 0),
        Point(6, 4),
        Point(8, 4),
        Point(6, 8),
        Point(0, 6),
        Point(0, 4),
        Point(2, 4),
    ]
    candidate = Point(6, 0)
    matching = Point(2, 4)
    assert find_blocked_rectangle(candidate, matching, polygon_points) == [
        [(Point(2, 4), Point(6, 4))]
    ]

    # Test case 2: Multiple blocked rectangles exist
    polygon_points = [
        Point(2, 0),
        Point(6, 0),
        Point(6, 4),
        Point(8, 4),
        Point(6, 8),
        Point(0, 6),
        Point(0, 4),
        Point(2, 4),
    ]
    candidate = Point(6, 0)
    matching = Point(2, 6)
    assert find_blocked_rectangle(candidate, matching, polygon_points) == [
        [(Point(2, 4), Point(6, 4))],
        [(Point(2, 6), Point(6, 6))],
    ]

    # Test case 3: Blocked rectangle does not exist
    polygon_points = [
        Point(1, 5),
        Point(1, 4),
        Point(3, 4),
        Point(3, 3),
        Point(2, 3),
        Point(2, 1),
        Point(5, 1),
        Point(8, 2),
        Point(8, 1),
        Point(9, 1),
        Point(9, 4),
        Point(8, 4),
        Point(8, 5),
    ]
    candidate = Point(8, 2)
    matching = Point(3, 5)
    assert find_blocked_rectangle(candidate, matching, polygon_points) == None

    # Test case 4: Empty polygon
    polygon_points = []
    candidate = Point(2, 0)
    matching = Point(6, 0)
    assert find_blocked_rectangle(candidate, matching, polygon_points) == None

    # Test case 5: Blocked rectangle on the boundary
    polygon_points = [
        Point(0, 0),
        Point(0, 3),
        Point(3, 3),
        Point(4, 3),
        Point(4, 2),
        Point(12, 2),
        Point(12, 10),
        Point(10, 10),
        Point(10, 8),
        Point(6, 8),
        Point(6, 14),
        Point(0, 14),
    ]
    candidate = Point(8, 2)
    matching = Point(3, 5)
    assert find_blocked_rectangle(candidate, matching, polygon_points) == [
        [(Point(3, 2), Point(3, 3))],
        [(Point(3, 2), Point(5, 2))],
        [(Point(8, 2), Point(8, 4))],
        [(Point(3, 4), Point(3, 5))],
    ]


test_find_blocked_rectangle()


def test_is_valid_rectilinear_polygon():
    # Test case 1: Valid rectilinear polygon
    polygon_points = [
        Point(2, 0),
        Point(6, 0),
        Point(6, 4),
        Point(8, 4),
        Point(6, 8),
        Point(0, 6),
        Point(0, 4),
        Point(2, 4),
    ]
    assert is_valid_rectilinear_polygon(polygon_points) == True

    # Test case 2: Valid rectilinear polygon
    polygon_points = [
        Point(1, 5),
        Point(1, 4),
        Point(3, 4),
        Point(3, 3),
        Point(2, 3),
        Point(2, 1),
        Point(5, 1),
        Point(8, 2),
        Point(8, 1),
        Point(9, 1),
        Point(9, 4),
        Point(8, 4),
        Point(8, 5),
    ]
    assert is_valid_rectilinear_polygon(polygon_points) == True

    # Test case 3: Invalid rectilinear polygon (not closed)
    polygon_points = [
        Point(6, 0),
        Point(4, 6),
        Point(4, 8),
        Point(6, 8),
        Point(0, 6),
        Point(0, 4),
        Point(4, 2),
    ]
    assert is_valid_rectilinear_polygon(polygon_points) == False

    # Test case 4: Invalid rectilinear polygon (not closed)
    polygon_points = [
        Point(1, 5),
        Point(1, 4),
        Point(3, 4),
        Point(3, 3),
        Point(2, 3),
        Point(2, 1),
        Point(5, 1),
        Point(8, 2),
        Point(8, 1),
        Point(9, 1),
        Point(9, 4),
        Point(8, 4),
    ]
    assert is_valid_rectilinear_polygon(polygon_points) == False

    # Test case 5: Invalid rectilinear polygon (not rectilinear)
    polygon_points = [Point(0, 0), Point(3, 0), Point(0, 6)]
    assert is_valid_rectilinear_polygon(polygon_points) == False

    # Test case 6: Invalid rectilinear polygon (not rectilinear)
    polygon_points = [
        Point(0, 0),
        Point(0, 3),
        Point(1, 3),
        Point(1, 1),
        Point(3, 1),
        Point(3, 0),
    ]
    assert is_valid_rectilinear_polygon(polygon_points) == False

    # Test case 7: Empty polygon
    polygon_points = []
    assert is_valid_rectilinear_polygon(polygon_points) == False


test_is_valid_rectilinear_polygon()
