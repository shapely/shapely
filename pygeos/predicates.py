import warnings

from . import Geometry  # NOQA
from . import lib
from .decorators import multithreading_enabled, requires_geos

__all__ = [
    "has_z",
    "is_ccw",
    "is_closed",
    "is_empty",
    "is_geometry",
    "is_missing",
    "is_prepared",
    "is_ring",
    "is_simple",
    "is_valid",
    "is_valid_input",
    "is_valid_reason",
    "crosses",
    "contains",
    "contains_properly",
    "covered_by",
    "covers",
    "disjoint",
    "equals",
    "intersects",
    "overlaps",
    "touches",
    "within",
    "equals_exact",
    "relate",
    "relate_pattern",
]


@multithreading_enabled
def has_z(geometry, **kwargs):
    """Returns True if a geometry has a Z coordinate.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> has_z(Geometry("POINT (0 0)"))
    False
    >>> has_z(Geometry("POINT Z (0 0 0)"))
    True
    """
    return lib.has_z(geometry, **kwargs)


@requires_geos("3.7.0")
@multithreading_enabled
def is_ccw(geometry, **kwargs):
    """Returns True if a linestring or linearring is counterclockwise.

    Note that there are no checks on whether lines are actually closed and
    not self-intersecting, while this is a requirement for is_ccw. The recommended
    usage of this function for linestrings is ``is_ccw(g) & is_simple(g)`` and for
    linearrings ``is_ccw(g) & is_valid(g)``.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for non-linear goemetries and for
        lines with fewer than 4 points (including the closing point).
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    is_simple : Checks if a linestring is closed and simple.
    is_valid : Checks additionally if the geometry is simple.

    Examples
    --------
    >>> is_ccw(Geometry("LINEARRING (0 0, 0 1, 1 1, 0 0)"))
    False
    >>> is_ccw(Geometry("LINEARRING (0 0, 1 1, 0 1, 0 0)"))
    True
    >>> is_ccw(Geometry("LINESTRING (0 0, 1 1, 0 1)"))
    False
    >>> is_ccw(Geometry("POINT (0 0)"))
    False
    """
    return lib.is_ccw(geometry, **kwargs)


@multithreading_enabled
def is_closed(geometry, **kwargs):
    """Returns True if a linestring's first and last points are equal.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for non-linestrings.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

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
    return lib.is_closed(geometry, **kwargs)


@multithreading_enabled
def is_empty(geometry, **kwargs):
    """Returns True if a geometry is an empty point, polygon, etc.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    is_missing : checks if the object is a geometry

    Examples
    --------
    >>> is_empty(Geometry("POINT EMPTY"))
    True
    >>> is_empty(Geometry("POINT (0 0)"))
    False
    >>> is_empty(None)
    False
    """
    return lib.is_empty(geometry, **kwargs)


@multithreading_enabled
def is_geometry(geometry, **kwargs):
    """Returns True if the object is a geometry

    Parameters
    ----------
    geometry : any object or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    is_missing : check if an object is missing (None)
    is_valid_input : check if an object is a geometry or None

    Examples
    --------
    >>> is_geometry(Geometry("POINT (0 0)"))
    True
    >>> is_geometry(Geometry("GEOMETRYCOLLECTION EMPTY"))
    True
    >>> is_geometry(None)
    False
    >>> is_geometry("text")
    False
    """
    return lib.is_geometry(geometry, **kwargs)


@multithreading_enabled
def is_missing(geometry, **kwargs):
    """Returns True if the object is not a geometry (None)

    Parameters
    ----------
    geometry : any object or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    is_geometry : check if an object is a geometry
    is_valid_input : check if an object is a geometry or None
    is_empty : checks if the object is an empty geometry

    Examples
    --------
    >>> is_missing(Geometry("POINT (0 0)"))
    False
    >>> is_missing(Geometry("GEOMETRYCOLLECTION EMPTY"))
    False
    >>> is_missing(None)
    True
    >>> is_missing("text")
    False
    """
    return lib.is_missing(geometry, **kwargs)


@multithreading_enabled
def is_prepared(geometry, **kwargs):
    """Returns True if a Geometry is prepared.

    Note that it is not necessary to check if a geometry is already prepared
    before preparing it. It is more efficient to call ``prepare`` directly
    because it will skip geometries that are already prepared.

    This function will return False for missing geometries (None).

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    is_valid_input : check if an object is a geometry or None
    prepare : prepare a geometry

    Examples
    --------
    >>> geometry = Geometry("POINT (0 0)")
    >>> is_prepared(Geometry("POINT (0 0)"))
    False
    >>> from pygeos import prepare; prepare(geometry);
    >>> is_prepared(geometry)
    True
    >>> is_prepared(None)
    False
    """
    return lib.is_prepared(geometry, **kwargs)


@multithreading_enabled
def is_valid_input(geometry, **kwargs):
    """Returns True if the object is a geometry or None

    Parameters
    ----------
    geometry : any object or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    is_geometry : checks if an object is a geometry
    is_missing : checks if an object is None

    Examples
    --------
    >>> is_valid_input(Geometry("POINT (0 0)"))
    True
    >>> is_valid_input(Geometry("GEOMETRYCOLLECTION EMPTY"))
    True
    >>> is_valid_input(None)
    True
    >>> is_valid_input(1.0)
    False
    >>> is_valid_input("text")
    False
    """
    return lib.is_valid_input(geometry, **kwargs)


@multithreading_enabled
def is_ring(geometry, **kwargs):
    """Returns True if a linestring is closed and simple.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for non-linestrings.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

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
    return lib.is_ring(geometry, **kwargs)


@multithreading_enabled
def is_simple(geometry, **kwargs):
    """Returns True if a Geometry has no anomalous geometric points, such as
    self-intersections or self tangency.

    Note that polygons and linearrings are assumed to be simple. Use is_valid
    to check these kind of geometries for self-intersections.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for geometrycollections.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    is_ring : Checks additionally if the geometry is closed.
    is_valid : Checks whether a geometry is well formed.

    Examples
    --------
    >>> is_simple(Geometry("POLYGON((1 1, 2 1, 2 2, 1 1))"))
    True
    >>> is_simple(Geometry("LINESTRING(0 0, 1 1, 0 1, 1 0, 0 0)"))
    False
    >>> is_simple(None)
    False
    """
    return lib.is_simple(geometry, **kwargs)


@multithreading_enabled
def is_valid(geometry, **kwargs):
    """Returns True if a geometry is well formed.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted. Returns False for missing values.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    is_valid_reason : Returns the reason in case of invalid.

    Examples
    --------
    >>> is_valid(Geometry("LINESTRING(0 0, 1 1)"))
    True
    >>> is_valid(Geometry("POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))"))
    False
    >>> is_valid(Geometry("GEOMETRYCOLLECTION EMPTY"))
    True
    >>> is_valid(None)
    False
    """
    # GEOS is valid will emit warnings for invalid geometries. Suppress them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = lib.is_valid(geometry, **kwargs)
    return result


def is_valid_reason(geometry, **kwargs):
    """Returns a string stating if a geometry is valid and if not, why.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted. Returns None for missing values.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    is_valid : returns True or False

    Examples
    --------
    >>> is_valid_reason(Geometry("LINESTRING(0 0, 1 1)"))
    'Valid Geometry'
    >>> is_valid_reason(Geometry("POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))"))
    'Self-intersection[0 0]'
    >>> is_valid_reason(None) is None
    True
    """
    return lib.is_valid_reason(geometry, **kwargs)


@multithreading_enabled
def crosses(a, b, **kwargs):
    """Returns True if A and B spatially cross.

    A crosses B if they have some but not all interior points in common,
    the intersection is one dimension less than the maximum dimension of A or B,
    and the intersection is not equal to either A or B.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> # A contains B:
    >>> crosses(line, Geometry("POINT (0.5 0.5)"))
    False
    >>> # A and B intersect at a point but do not share all points:
    >>> crosses(line, Geometry("MULTIPOINT ((0 1), (0.5 0.5))"))
    True
    >>> crosses(line, Geometry("LINESTRING(0 1, 1 0)"))
    True
    >>> # A is contained by B; their intersection is a line (same dimension):
    >>> crosses(line, Geometry("LINESTRING(0 0, 2 2)"))
    False
    >>> area = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> # A contains B:
    >>> crosses(area, line)
    False
    >>> # A and B intersect with a line (lower dimension) but do not share all points:
    >>> crosses(area, Geometry("LINESTRING(0 0, 2 2)"))
    True
    >>> # A contains B:
    >>> crosses(area, Geometry("POINT (0.5 0.5)"))
    False
    >>> # A contains some but not all points of B; they intersect at a point:
    >>> crosses(area, Geometry("MULTIPOINT ((2 2), (0.5 0.5))"))
    True
    """
    return lib.crosses(a, b, **kwargs)


@multithreading_enabled
def contains(a, b, **kwargs):
    """Returns True if geometry B is completely inside geometry A.

    A contains B if no points of B lie in the exterior of A and at least one
    point of the interior of B lies in the interior of A.

    Note: following this definition, a geometry does not contain its boundary,
    but it does contain itself. See ``contains_properly`` for a version where
    a geometry does not contain itself.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    within : ``contains(A, B) == within(B, A)``
    contains_properly : contains with no common boundary points
    prepare : improve performance by preparing ``a`` (the first argument)

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
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))")
    >>> contains(polygon_with_hole, Geometry("POINT(1 1)"))
    True
    >>> contains(polygon_with_hole, Geometry("POINT(2 2)"))
    False
    >>> contains(polygon_with_hole, Geometry("LINESTRING(1 1, 5 5)"))
    False
    >>> contains(area, area)
    True
    >>> contains(area, None)
    False
    """
    return lib.contains(a, b, **kwargs)


@multithreading_enabled
def contains_properly(a, b, **kwargs):
    """Returns True if geometry B is completely inside geometry A, with no
    common boundary points.

    A contains B properly if B intersects the interior of A but not the
    boundary (or exterior). This means that a geometry A does not
    "contain properly" itself, which contrasts with the ``contains`` function,
    where common points on the boundary are allowed.

    Note: this function will prepare the geometries under the hood if needed.
    You can prepare the geometries in advance to avoid repeated preparation
    when calling this function multiple times.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    contains : contains which allows common boundary points
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> area1 = Geometry("POLYGON((0 0, 3 0, 3 3, 0 3, 0 0))")
    >>> area2 = Geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> area3 = Geometry("POLYGON((1 1, 2 1, 2 2, 1 2, 1 1))")

    ``area1`` and ``area2`` have a common border:

    >>> contains(area1, area2)
    True
    >>> contains_properly(area1, area2)
    False

    ``area3`` is completely inside ``area1`` with no common border:

    >>> contains(area1, area3)
    True
    >>> contains_properly(area1, area3)
    True
    """
    return lib.contains_properly(a, b, **kwargs)


@multithreading_enabled
def covered_by(a, b, **kwargs):
    """Returns True if no point in geometry A is outside geometry B.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    covers : ``covered_by(A, B) == covers(B, A)``
    prepare : improve performance by preparing ``a`` (the first argument)

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
    >>> covered_by(None, area)
    False
    """
    return lib.covered_by(a, b, **kwargs)


@multithreading_enabled
def covers(a, b, **kwargs):
    """Returns True if no point in geometry B is outside geometry A.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    covered_by : ``covers(A, B) == covered_by(B, A)``
    prepare : improve performance by preparing ``a`` (the first argument)

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
    >>> covers(area, None)
    False
    """
    return lib.covers(a, b, **kwargs)


@multithreading_enabled
def disjoint(a, b, **kwargs):
    """Returns True if A and B do not share any point in space.

    Disjoint implies that overlaps, touches, within, and intersects are False.
    Note missing (None) values are never disjoint.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    intersects : ``disjoint(A, B) == ~intersects(A, B)``
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> disjoint(line, Geometry("POINT (0 0)"))
    False
    >>> disjoint(line, Geometry("POINT (0 1)"))
    True
    >>> disjoint(line, Geometry("LINESTRING(0 2, 2 0)"))
    False
    >>> empty = Geometry("GEOMETRYCOLLECTION EMPTY")
    >>> disjoint(line, empty)
    True
    >>> disjoint(empty, empty)
    True
    >>> disjoint(empty, None)
    False
    >>> disjoint(None, None)
    False
    """
    return lib.disjoint(a, b, **kwargs)


@multithreading_enabled
def equals(a, b, **kwargs):
    """Returns True if A and B are spatially equal.

    If A is within B and B is within A, A and B are considered equal. The
    ordering of points can be different.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See Also
    --------
    equals_exact : Check if A and B are structurally equal given a specified
        tolerance.

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 5 5, 10 10)")
    >>> equals(line, Geometry("LINESTRING(0 0, 10 10)"))
    True
    >>> equals(Geometry("POLYGON EMPTY"), Geometry("GEOMETRYCOLLECTION EMPTY"))
    True
    >>> equals(None, None)
    False
    """
    return lib.equals(a, b, **kwargs)


@multithreading_enabled
def intersects(a, b, **kwargs):
    """Returns True if A and B share any portion of space.

    Intersects implies that overlaps, touches and within are True.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    disjoint : ``intersects(A, B) == ~disjoint(A, B)``
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> intersects(line, Geometry("POINT (0 0)"))
    True
    >>> intersects(line, Geometry("POINT (0 1)"))
    False
    >>> intersects(line, Geometry("LINESTRING(0 2, 2 0)"))
    True
    >>> intersects(None, None)
    False
    """
    return lib.intersects(a, b, **kwargs)


@multithreading_enabled
def overlaps(a, b, **kwargs):
    """Returns True if A and B spatially overlap.

    A and B overlap if they have some but not all points in common, have the
    same dimension, and the intersection of the interiors of the two geometries
    has the same dimension as the geometries themselves.  That is, only polyons
    can overlap other polygons and only lines can overlap other lines.

    If either A or B are None, the output is always False.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    prepare : improve performance by preparing ``a`` (the first argument)

    Examples
    --------
    >>> poly = Geometry("POLYGON ((0 0, 0 4, 4 4, 4 0, 0 0))")
    >>> # A and B share all points (are spatially equal):
    >>> overlaps(poly, poly)
    False
    >>> # A contains B; all points of B are within A:
    >>> overlaps(poly, Geometry("POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))"))
    False
    >>> # A partially overlaps with B:
    >>> overlaps(poly, Geometry("POLYGON ((2 2, 2 6, 6 6, 6 2, 2 2))"))
    True
    >>> line = Geometry("LINESTRING (2 2, 6 6)")
    >>> # A and B are different dimensions; they cannot overlap:
    >>> overlaps(poly, line)
    False
    >>> overlaps(poly, Geometry("POINT (2 2)"))
    False
    >>> # A and B share some but not all points:
    >>> overlaps(line, Geometry("LINESTRING (0 0, 4 4)"))
    True
    >>> # A and B intersect only at a point (lower dimension); they do not overlap
    >>> overlaps(line, Geometry("LINESTRING (6 0, 0 6)"))
    False
    >>> overlaps(poly, None)
    False
    >>> overlaps(None, None)
    False
    """
    return lib.overlaps(a, b, **kwargs)


@multithreading_enabled
def touches(a, b, **kwargs):
    """Returns True if the only points shared between A and B are on the
    boundary of A and B.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    prepare : improve performance by preparing ``a`` (the first argument)

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
    return lib.touches(a, b, **kwargs)


@multithreading_enabled
def within(a, b, **kwargs):
    """Returns True if geometry A is completely inside geometry B.

    A is within B if no points of A lie in the exterior of B and at least one
    point of the interior of A lies in the interior of B.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    contains : ``within(A, B) == contains(B, A)``
    prepare : improve performance by preparing ``a`` (the first argument)

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
    >>> within(None, area)
    False
    """
    return lib.within(a, b, **kwargs)


@multithreading_enabled
def equals_exact(a, b, tolerance=0.0, **kwargs):
    """Returns True if A and B are structurally equal.

    This method uses exact coordinate equality, which requires coordinates
    to be equal (within specified tolerance) and and in the same order for all
    components of a geometry. This is in contrast with the ``equals`` function
    which uses spatial (topological) equality.

    Parameters
    ----------
    a, b : Geometry or array_like
    tolerance : float or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See Also
    --------
    equals : Check if A and B are spatially equal.

    Examples
    --------
    >>> point1 = Geometry("POINT(50 50)")
    >>> point2 = Geometry("POINT(50.1 50.1)")
    >>> equals_exact(point1, point2)
    False
    >>> equals_exact(point1, point2, tolerance=0.2)
    True
    >>> equals_exact(point1, None, tolerance=0.2)
    False

    Difference between structucal and spatial equality:

    >>> polygon1 = Geometry("POLYGON((0 0, 1 1, 0 1, 0 0))")
    >>> polygon2 = Geometry("POLYGON((0 0, 0 1, 1 1, 0 0))")
    >>> equals_exact(polygon1, polygon2)
    False
    >>> equals(polygon1, polygon2)
    True
    """
    return lib.equals_exact(a, b, tolerance, **kwargs)


def relate(a, b, **kwargs):
    """
    Returns a string representation of the DE-9IM intersection matrix.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> point = Geometry("POINT (0 0)")
    >>> line = Geometry("LINESTRING(0 0, 1 1)")
    >>> relate(point, line)
    'F0FFFF102'
    """
    return lib.relate(a, b, **kwargs)


@multithreading_enabled
def relate_pattern(a, b, pattern, **kwargs):
    """
    Returns True if the DE-9IM string code for the relationship between
    the geometries satisfies the pattern, else False.

    This function compares the DE-9IM code string for two geometries
    against a specified pattern. If the string matches the pattern then
    ``True`` is returned, otherwise ``False``. The pattern specified can
    be an exact match (``0``, ``1`` or ``2``), a boolean match
    (uppercase ``T`` or ``F``), or a wildcard (``*``). For example,
    the pattern for the ``within`` predicate is ``'T*F**F***'``.

    Parameters
    ----------
    a, b : Geometry or array_like
    pattern : string
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> point = Geometry("POINT (0.5 0.5)")
    >>> square = Geometry("POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))")
    >>> relate(point, square)
    '0FFFFF212'
    >>> relate_pattern(point, square, "T*F**F***")
    True
    """
    return lib.relate_pattern(a, b, pattern, **kwargs)
