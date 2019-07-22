import warnings

from . import ufuncs
from .geometry import Geometry
from .ufuncs import Empty  # NOQA

__all__ = [
    "is_closed",
    "is_empty",
    "is_ring",
    "is_simple",
    "is_valid",
    "is_valid_reason",
    "crosses",
    "contains",
    "covered_by",
    "covers",
    "disjoint",
    "equals",
    "intersects",
    "overlaps",
    "touches",
    "within",
]


def is_closed(geometry, **kwargs):
    """Returns True if a linestring's first and last points are equal.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for non-linestrings.

    See also
    --------
    is_ring : Checks additionally if the geometry is simple.

    Examples
    --------
    >>> is_closed(Geometry("LINESTRING (0 0, 1 1)"))
    False
    >>> is_closed(Geometry("LINESTRING(0 0, 0 1, 1 1, 0 0)"))
    True
    >>> is_closed(Geometry("POINT (0 0)"))
    False
    """
    return ufuncs.is_closed(geometry, **kwargs)


def is_empty(geometry, **kwargs):
    """Returns True if a geometry is an empty point, polygon, etc.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted.

    Examples
    --------
    >>> is_empty(Geometry("POINT EMPTY"))
    True
    >>> is_empty(Geometry("POINT (0 0)"))
    False
    >>> is_empty(Empty)
    True
    """
    return ufuncs.is_empty(geometry, **kwargs)


def is_ring(geometry, **kwargs):
    """Returns True if a linestring is closed and simple.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for non-linestrings.

    See also
    --------
    is_closed : Checks only if the geometry is closed.
    is_simple : Checks only if the geometry is simple.

    Examples
    --------
    >>> is_ring(Geometry("POINT (0 0)"))
    False
    >>> geom = Geometry("LINESTRING(0 0, 1 1)")
    >>> is_closed(geom), is_simple(geom), is_ring(geom)
    (False, True, False)
    >>> geom = Geometry("LINESTRING(0 0, 0 1, 1 1, 0 0)")
    >>> is_closed(geom), is_simple(geom), is_ring(geom)
    (True, True, True)
    >>> geom = Geometry("LINESTRING(0 0, 1 1, 0 1, 1 0, 0 0)")
    >>> is_closed(geom), is_simple(geom), is_ring(geom)
    (True, False, False)
    """
    return ufuncs.is_ring(geometry, **kwargs)


def is_simple(geometry, **kwargs):
    """Returns True if a Geometry has no anomalous geometric points, such as
    self-intersections or self tangency.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for geometrycollections.

    See also
    --------
    is_ring : Checks additionally if the geometry is closed.

    Examples
    --------
    >>> is_simple(Geometry("POLYGON((1 2, 3 4, 5 6, 1 2))"))
    True
    >>> is_simple(Geometry("LINESTRING(0 0, 1 1, 0 1, 1 0, 0 0)"))
    False
    >>> is_simple(Empty)
    False
    """
    return ufuncs.is_simple(geometry, **kwargs)


def is_valid(geometry, **kwargs):
    """Returns True if a geometry is well formed.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted.

    See also
    --------
    is_valid_reason : Returns the reason in case of invalid.

    Examples
    --------
    >>> is_valid(Geometry("LINESTRING(0 0, 1 1)"))
    True
    >>> is_valid(Geometry("POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))"))
    False
    >>> is_valid(Empty)
    True
    """
    # GEOS is valid will emit warnings for invalid geometries. Suppress them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ufuncs.is_valid(geometry, **kwargs)
    return result


def is_valid_reason(geometry, **kwargs):
    """Returns a string stating if a geometry is valid and if not, why.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted.

    See also
    --------
    is_valid : returns True or False

    Examples
    --------
    >>> is_valid_reason(Geometry("LINESTRING(0 0, 1 1)"))
    'Valid Geometry'
    >>> is_valid_reason(Geometry("POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))"))
    'Self-intersection[0 0]'
    """
    return ufuncs.is_valid_reason(geometry, **kwargs)


def crosses(a, b, **kwargs):
    """Returns True if the intersection of two geometries spatially crosses.

    That is: the geometries have some, but not all interior points in common.
    The geometries must intersect and the intersection must have a
    dimensionality less than the maximum dimension of the two input geometries.
    Additionally, the intersection of the two geometries must not equal either
    of the source geometries.

    Parameters
    ----------
    a, b : Geometry or array_like

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> crosses(line, Geometry("POINT (0.5 0.5)"))
    False
    >>> crosses(line, Geometry("MULTIPOINT ((0 1), (0.5 0.5))"))
    True
    >>> crosses(line, Geometry("LINESTRING(0 1, 1 0)"))
    True
    >>> crosses(line, Geometry("LINESTRING(0 0, 2 2)"))
    False
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> crosses(area, line)
    False
    >>> crosses(area, Geometry("LINESTRING(0 0, 2 2)"))
    True
    >>> crosses(area, Geometry("POINT (0.5 0.5)"))
    False
    >>> crosses(area, Geometry("MULTIPOINT ((2 2), (0.5 0.5))"))
    True
    """
    return ufuncs.crosses(a, b, **kwargs)


def contains(a, b, **kwargs):
    """Returns True if geometry B is completely inside geometry A.

    A contains B if no points of B lie in the exterior of A and at least one
    point of the interior of B lies in the interior of A.

    Parameters
    ----------
    a, b : Geometry or array_like

    See also
    --------
    within : ``contains(A, B) == within(B, A)``

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> contains(line, Geometry("POINT (0 0)"))
    False
    >>> contains(line, Geometry("POINT (0.5 0.5)"))
    True
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> contains(area, Geometry("POINT (0 0)"))
    False
    >>> contains(area, line)
    True
    >>> contains(area, Geometry("LINESTRING(0 0, 2 2)"))
    False
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))")  # NOQA
    >>> contains(polygon_with_hole, Geometry("POINT(1 1)"))
    True
    >>> contains(polygon_with_hole, Geometry("POINT(2 2)"))
    False
    >>> contains(polygon_with_hole, Geometry("LINESTRING(1 1, 5 5)"))
    False
    >>> contains(area, area)
    True
    >>> contains(area, Empty)
    False
    """
    return ufuncs.contains(a, b, **kwargs)


def covered_by(a, b, **kwargs):
    """Returns True if no point in geometry A is outside geometry B.

    Parameters
    ----------
    a, b : Geometry or array_like

    See also
    --------
    covers : ``covered_by(A, B) == covers(B, A)``

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> covered_by(Geometry("POINT (0 0)"), line)
    True
    >>> covered_by(Geometry("POINT (0.5 0.5)"), line)
    True
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> covered_by(Geometry("POINT (0 0)"), area)
    True
    >>> covered_by(line, area)
    True
    >>> covered_by(Geometry("LINESTRING(0 0, 2 2)"), area)
    False
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))")  # NOQA
    >>> covered_by(Geometry("POINT(1 1)"), polygon_with_hole)
    True
    >>> covered_by(Geometry("POINT(2 2)"), polygon_with_hole)
    True
    >>> covered_by(Geometry("LINESTRING(1 1, 5 5)"), polygon_with_hole)
    False
    >>> covered_by(area, area)
    True
    >>> covered_by(Empty, area)
    False
    """
    return ufuncs.covered_by(a, b, **kwargs)


def covers(a, b, **kwargs):
    """Returns True if no point in geometry B is outside geometry A.

    Parameters
    ----------
    a, b : Geometry or array_like

    See also
    --------
    covered_by : ``covers(A, B) == covered_by(B, A)``

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> covers(line, Geometry("POINT (0 0)"))
    True
    >>> covers(line, Geometry("POINT (0.5 0.5)"))
    True
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> covers(area, Geometry("POINT (0 0)"))
    True
    >>> covers(area, line)
    True
    >>> covers(area, Geometry("LINESTRING(0 0, 2 2)"))
    False
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))")  # NOQA
    >>> covers(polygon_with_hole, Geometry("POINT(1 1)"))
    True
    >>> covers(polygon_with_hole, Geometry("POINT(2 2)"))
    True
    >>> covers(polygon_with_hole, Geometry("LINESTRING(1 1, 5 5)"))
    False
    >>> covers(area, area)
    True
    >>> covers(area, Empty)
    False
    """
    return ufuncs.covers(a, b, **kwargs)


def disjoint(a, b, **kwargs):
    """Returns True if A and B do not share any point in space.

    Disjoint implies that overlaps, touches, within, and intersects are False.

    Parameters
    ----------
    a, b : Geometry or array_like

    See also
    --------
    intersects : ``disjoint(A, B) == ~intersects(A, B)``

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> disjoint(line, Geometry("POINT (0 0)"))
    False
    >>> disjoint(line, Geometry("POINT (0 1)"))
    True
    >>> disjoint(line, Geometry("LINESTRING(0 2, 2 0)"))
    False
    >>> disjoint(Empty, Empty)
    True
    """
    return ufuncs.disjoint(a, b, **kwargs)


def equals(a, b, tolerance=0.0, **kwargs):
    """Returns True if A and B are spatially equal.

    If A is within B and B is within A, A and B are considered equal. The
    ordering of points can be different. Optionally, a tolerance can be
    provided for comparing vertices.

    Parameters
    ----------
    a, b : Geometry or array_like
    tolerance : float or array_like


    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 5 5, 10 10)")
    >>> equals(line, Geometry("LINESTRING(0 0, 10 10)"))
    True
    >>> equals(Geometry("POINT (5 5)"), Geometry("POINT (5.1 5)"), tolerance=0.1)
    True
    >>> equals(Empty, Empty)
    True
    """
    if tolerance > 0.0:
        return ufuncs.equals_exact(a, b, tolerance, **kwargs)
    else:
        return ufuncs.equals(a, b, **kwargs)


def intersects(a, b, **kwargs):
    """Returns True if A and B share any portion of space.

    Intersects implies that overlaps, touches and within are True.

    Parameters
    ----------
    a, b : Geometry or array_like

    See also
    --------
    disjoint : ``intersects(A, B) == ~disjoint(A, B)``

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> intersects(line, Geometry("POINT (0 0)"))
    True
    >>> intersects(line, Geometry("POINT (0 1)"))
    False
    >>> intersects(line, Geometry("LINESTRING(0 2, 2 0)"))
    True
    >>> intersects(Empty, Empty)
    False
    """
    return ufuncs.intersects(a, b, **kwargs)


def overlaps(a, b, **kwargs):
    """Returns True if A and B intersect, but one does not completely contain
    the other.

    Parameters
    ----------
    a, b : Geometry or array_like

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> overlaps(line, line)
    False
    >>> overlaps(line, Geometry("LINESTRING(0 0, 2 2)"))
    False
    >>> overlaps(line, Geometry("LINESTRING(0.5 0.5, 2 2)"))
    True
    >>> overlaps(line, Geometry("POINT (0.5 0.5)"))
    False
    >>> overlaps(Empty, Empty)
    False
    """
    return ufuncs.overlaps(a, b, **kwargs)


def touches(a, b, **kwargs):
    """Returns True if the only points shared between A and B are on the
    boundary of A and B.

    Parameters
    ----------
    a, b: Geometry or array_like

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 2, 2 0)")
    >>> touches(line, Geometry("POINT(0 2)"))
    True
    >>> touches(line, Geometry("POINT(1 1)"))
    False
    >>> touches(line, Geometry("LINESTRING(0 0, 1 1)"))
    True
    >>> touches(line, Geometry("LINESTRING(0 0, 2 2)"))
    False
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> touches(area, Geometry("POINT(0.5 0)"))
    True
    >>> touches(area, Geometry("POINT(0.5 0.5)"))
    False
    >>> touches(area, line)
    True
    >>> touches(area, Geometry("POLYGON((0 1, 1 1, 1 2, 0 2, 0 1))"))
    True
    """
    return ufuncs.touches(a, b, **kwargs)


def within(a, b, **kwargs):
    """Returns True if geometry A is completely inside geometry B.

    A is within B if no points of A lie in the exterior of B and at least one
    point of the interior of A lies in the interior of B.

    Parameters
    ----------
    a, b : Geometry or array_like

    See also
    --------
    contains : ``within(A, B) == contains(B, A)``

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> within(Geometry("POINT (0 0)"), line)
    False
    >>> within(Geometry("POINT (0.5 0.5)"), line)
    True
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> within(Geometry("POINT (0 0)"), area)
    False
    >>> within(line, area)
    True
    >>> within(Geometry("LINESTRING(0 0, 2 2)"), area)
    False
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))")  # NOQA
    >>> within(Geometry("POINT(1 1)"), polygon_with_hole)
    True
    >>> within(Geometry("POINT(2 2)"), polygon_with_hole)
    False
    >>> within(Geometry("LINESTRING(1 1, 5 5)"), polygon_with_hole)
    False
    >>> within(area, area)
    True
    >>> within(Empty, area)
    False
    """
    return ufuncs.within(a, b, **kwargs)
