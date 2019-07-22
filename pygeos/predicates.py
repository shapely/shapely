import warnings

from . import ufuncs
from .geometry import Geometry

__all__ = [
    "is_closed",
    "is_empty",
    "is_ring",
    "is_simple",
    "is_valid",
    "crosses",
    "contains",
    "covered_by",
    "covers",
    "disjoint",
    "equals",
    "equals_exact",
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
    - is_valid : returns True or False

    Examples
    --------
    >>> is_valid_reason(Geometry("LINESTRING(0 0, 1 1)"))
    'Valid Geometry'
    >>> is_valid_reason(Geometry("POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))"))
    'Self-intersection[0 0]'
    """
    return ufuncs.is_valid_reason(geometry, **kwargs)


def crosses(a, b):
    """Returns True if the intersection of two geometries spatially crosses.

    That is: the geometries have some, but not all interior points in common.
    The geometries must intersect and the intersection must have a
    dimensionality less than the maximum dimension of the two input geometries.
    Additionally, the intersection of the two geometries must not equal either
    of the source geometries.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like

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
    return ufuncs.crosses(a, b)


def contains(a, b):
    """Returns True when geometry B contains geometry A.

    A contains B if no points of B lie in the exterior of A and at  least one
    point of the interior of B lies in the interior of A.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like

    See also
    --------
    - within : Contains is the inverse of within (except when dealing with
      invalid geometries).

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> contains(line, Geometry("POINT (0.5 0.5)"))
    True
    >>> contains(line, Geometry("POINT (0 0)"))
    True
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> contains(area, Geometry("POINT (0 0)"))
    False
    >>> contains(area, Geometry("POINT (0.5 0.5)"))
    True
    >>> contains(area, Geometry("POINT (0.5 0.5)"))
    True
    """
    return ufuncs.contains(a, b)


def covered_by(a, b):
    return ufuncs.covered_by(a, b)


def covers(a, b):
    return ufuncs.covers(a, b)


def disjoint(a, b):
    return ufuncs.disjoint(a, b)


def equals(a, b):
    return ufuncs.equals(a, b)


def equals_exact(a, b, tolerance):
    return ufuncs.equals_exact(a, b, tolerance)


def intersects(a, b):
    return ufuncs.intersects(a, b)


def overlaps(a, b):
    return ufuncs.overlaps(a, b)


def touches(a, b):
    return ufuncs.touches(a, b)


def within(a, b):
    return ufuncs.within(a, b)
