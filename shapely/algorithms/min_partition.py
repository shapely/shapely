"""
An implementation of the algorithms in:
"Minimum edge length partitioning of rectilinear polygons", by Lingas, Pinter, Rivest and Shamir from 1982
https://people.csail.mit.edu/rivest/pubs/LPRS82.pdf
Programmer: Dvir Borochov
Date: 20-04-24 
"""

from shapely.geometry import Polygon
from shapely.geometry.point import Point


def minimum_edge_length_partition(polygon: Polygon):
    """
    The main algorithm.
    Performs minimum edge length partitioning of a rectilinear polygon into rectangles.

    Args:
        polygon (Polygon): A Shapely Polygon instance representing the rectilinear polygon.

    Returns:
        list of list of tuple: A list where each element is a list of tuples representing the vertices of the rectangles
                               resulting from the partition, or None if not found (in case the polygon isn't rectilinear).

    Examples:

        >>> minimum_edge_length_partition(Polygon([(1,5),(1,4),(3,4),(3,3),(2,3),(2,1),(5,1),(8,2),(8,1),(9,1),(9,4),(8,4),(8,5)]))
        [[(3,2), (3,3)], [(3,2), (5,2)], [(8,2), (8,4)], [(3,4), (3,5)], [(3,1), (3,2)]]

        >>> minimum_edge_length_partition(Polygon([(2,0),(6,0), (6,4), (8,4), (8,6),(0,6),(0,4), (2,4)]))
        [[(2,4), (6,4)]]

        >>> minimum_edge_length_partition(Polygon([(0,0),(0,3),(3,3),(4,3),(4,2),(12,2),(12,10),(10,10),(10,8),(6,8),(6,14),(0,14)]))
        [[(0,3), (3,3)], [(4,3), (4,8)], [(0,8), (6,8)], [(10,8),(8,2)], [(10,8),(12,10)]]
    """
    pass


def find_convex_points(polygon: Polygon):
    """
    Finds the convex points in a given rectilinear polygon.

    Args:
        polygon (Polygon): A Shapely Polygon instance representing the rectilinear polygon.

    Returns:
        list: List of Points representing the convex points of the polygon.

    Examples:
        >>> find_convex_points(Polygon([(1, 1), (1, 5), (5, 5), (5, 1)]))
        [Point(1, 1), Point(1, 5), Point(5, 5), Point(5, 1)]

        >>> find_convex_points(Polygon([(2,0),(6,0), (6,4), (8,4), (8,6), (0,6), (0,4), (2,4)]))
        [Point(2, 0), Point(6, 0), Point(6, 4), Point(8, 4), Point(8, 6), Point(0, 6), Point(0, 4), Point(4, 2)]

        >>> find_convex_points(Polygon([(0, 0), (0, 3), (1, 3), (1, 1), (3, 1), (3, 0)]))
        [Point(0, 0), Point(0, 3), Point(1, 3), Point(1, 1), Point(3, 1), Point(3, 0)]
    """
    convex_points = []
    for point in polygon.exterior.coords[
        :-1
    ]:  # Skip the last point because it duplicates the first one
        convex_points.append(Point(point))
    return convex_points


def find_matching_point(candidate: Point, polygon: Polygon):
    """
    Finds a matching point on the induced grid with respect to a given candidate point in a given polygon.
    The candidate point should be kitty-corner to the matching point in a blocked rectangle inside the polygon.

    Args:
        candidate (Point): The candidate point as a Shapely Point.
        polygon (Polygon): A Shapely Polygon instance representing the vertices of the polygon.

    Returns:
        list: List of Points that contains the matching point, or None if not found.

    Examples:
        >>> find_matching_point(Point(2, 0), Polygon([(2, 0), (6, 0), (6, 4), (8, 4), (8, 6), (0, 6), (0, 4), (4, 2)]))
        [Point(2, 4), Point(2, 6)]

        >>> find_matching_point(Point(8, 2), Polygon([(1, 5), (1, 4), (3, 4), (3, 3), (2, 3), (2, 1), (5, 1), (8, 2), (8, 1), (9, 1), (9, 4), (8, 4), (8, 5)]))
        [Point(3, 5), Point(3, 4), Point(3, 3)]
    """
    pass


def find_blocked_rectangle(candidate: Point, matching: Point, polygon: Polygon):
    """
    Finds a blocked rectangle with respect to two points in a given polygon.

    Args:
        candidate (Point): The first point as a Shapely Point which is a candidate point.
        matching (Point): The second point as a Shapely Point which is a matching point in relation to the candidate point.
        polygon (Polygon): A Shapely Polygon instance representing the vertices of the polygon.

    Returns:
        list of list of Point: representing the new edges of the blocked rectangle, or None if not found.

    Examples:
        >>> find_blocked_rectangle(Point(6, 0), Point(2, 4), Polygon([(2, 0), (6, 0), (6, 4), (8, 4), (8, 6), (0, 6), (0, 4), (4, 2)]))
        [[Point(2, 4), Point(6, 4)]]

        >>> find_blocked_rectangle(Point(6, 0), Point(2, 6), Polygon([(2, 0), (6, 0), (4, 6), (4, 8), (6, 8), (0, 6), (0, 4), (4, 2)]))
        [[Point(2, 4), Point(6, 4)], [Point(2, 6), Point(6, 6)]]

        >>> find_blocked_rectangle(Point(8, 2), Point(3, 5), Polygon([(1, 5), (1, 4), (3, 4), (3, 3), (2, 3), (2, 1), (5, 1), (8, 2), (8, 1), (9, 1), (9, 4), (8, 4), (8, 5)]))
        [[Point(3, 2), Point(3, 3)], [Point(3, 2), Point(5, 2)], [Point(8, 2), Point(8, 4)], [Point(3, 4), Point(3, 5)]]
    """
    # Placeholder for actual algorithm

    return []  # Placeholder return value


def is_valid_rectilinear_polygon(polygon: Polygon):
    """
    Validates whether the given polygon is a closed rectilinear polygon.

    Args:
        polygon (Polygon): A Shapely Polygon instance.

    Returns:
        bool: True if the polygon is a valid closed rectilinear polygon, False otherwise.

    Examples:
        >>> is_valid_rectilinear_polygon(Polygon([(2, 0), (6, 0), (6, 4), (8, 4), (8, 6), (0, 6), (0, 4), (4, 2)]))
        True

        >>> is_valid_rectilinear_polygon(Polygon([(1, 5), (1, 4), (3, 4), (3, 3), (2, 3), (2, 1), (5, 1), (8, 2), (8, 1), (9, 1), (9, 4), (8, 4), (8, 5)]))
        True

        >>> is_valid_rectilinear_polygon(Polygon([(6, 0), (4, 6), (4, 8), (6, 8), (0, 6), (0, 4), (4, 2)]))
        False

        >>> is_valid_rectilinear_polygon(Polygon([(1, 5), (1, 4), (3, 4), (3, 3), (2, 3), (2, 1), (5, 1), (8, 2), (8, 1), (9, 1), (9, 4), (8, 4)]))
        False
    """
    # Check if all edges are either horizontal or vertical
    coords = list(polygon.exterior.coords)
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        if not (x1 == x2 or y1 == y2):
            return False
    return True
