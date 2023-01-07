from typing import TYPE_CHECKING, Union

from shapely import lib
from shapely.decorators import multithreading_enabled
from shapely.errors import UnsupportedGEOSVersionError

__all__ = [
    "line_interpolate_point",
    "line_locate_point",
    "line_merge",
    "shared_paths",
    "shortest_line",
]

from shapely.shapely_typing import MaybeArrayN, MaybeArrayNLike, MaybeGeometryArrayNLike

if TYPE_CHECKING:
    from shapely import GeometryCollection, LineString, MultiLineString, Point


@multithreading_enabled
def line_interpolate_point(
    line: MaybeArrayNLike["LineString"],
    distance: MaybeArrayNLike[float],
    normalized: bool = False,
    **kwargs
) -> MaybeArrayN["Point"]:
    """Returns a point interpolated at given distance on a line.
    (a point at the specified distance along a linear geometry)

    Parameters
    ----------
    line : MaybeGeometryArrayNLike
        For multilinestrings or geometrycollections, the first geometry is taken
        and the rest is ignored. This function raises a TypeError for non-linear
        geometries. For empty linear geometries, empty points are returned.
    distance : MaybeArrayNLike[float]
        Negative values measure distance from the end of the line.
        Out-of-range values will be clipped to the line endings.
    normalized : bool, default False
        If True, the distance is a fraction of the total
        line length instead of the absolute distance.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import LineString
    >>> line = LineString([(0, 2), (0, 10)])
    >>> line_interpolate_point(line, 2)
    <POINT (0 4)>
    >>> line_interpolate_point(line, 100)
    <POINT (0 10)>
    >>> line_interpolate_point(line, -2)
    <POINT (0 8)>
    >>> line_interpolate_point(line, [0.25, -0.25], normalized=True).tolist()
    [<POINT (0 4)>, <POINT (0 8)>]
    >>> line_interpolate_point(LineString(), 1)
    <POINT EMPTY>
    """
    if normalized:
        return lib.line_interpolate_point_normalized(line, distance, **kwargs)
    else:
        return lib.line_interpolate_point(line, distance, **kwargs)


@multithreading_enabled
def line_locate_point(
    line: MaybeArrayNLike["LineString"],
    other: MaybeArrayNLike["Point"],
    normalized: bool = False,
    **kwargs
) -> MaybeArrayN[float]:
    """Returns the distance to the line origin of given point.
    (distance along this geometry to a point nearest the given point)

    If given point does not intersect with the line, the point will first be
    projected onto the line after which the distance is taken.

    Parameters
    ----------
    line : MaybeArrayNLike["LineString"]
    other : MaybeArrayNLike["Point"]
    normalized : bool, default False
        If True, the distance is a fraction of the total
        line length instead of the absolute distance.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> line = LineString([(0, 2), (0, 10)])
    >>> point = Point(4, 4)
    >>> line_locate_point(line, point)
    2.0
    >>> line_locate_point(line, point, normalized=True)
    0.25
    >>> line_locate_point(line, Point(0, 18))
    8.0
    >>> line_locate_point(LineString(), point)
    nan
    """
    if normalized:
        return lib.line_locate_point_normalized(line, other, **kwargs)
    else:
        return lib.line_locate_point(line, other, **kwargs)


@multithreading_enabled
def line_merge(
    line: MaybeArrayNLike["MultiLineString"], directed: bool = False, **kwargs
) -> MaybeArrayN[Union["LineString", "MultiLineString", "GeometryCollection"]]:
    """Returns (Multi)LineStrings formed by combining all connected lines in a
    MultiLineString.

    Lines are joined together at their endpoints in case two lines are
    intersecting. Lines are not joined when 3 or more lines are intersecting at
    the endpoints. Line elements that cannot be joined are kept as is in the
    resulting MultiLineString.

    The direction of each merged LineString will be that of the majority of the
    LineStrings from which it was derived. Except if ``directed=True`` is
    specified, then the operation will not change the order of points within
    lines and so only lines which can be joined with no change in direction
    are merged.

    Parameters
    ----------
    line : MaybeArrayNLike["MultiLineString"]
    directed : bool, default False
        Only combine lines if possible without changing point order.
        Requires GEOS >= 3.11.0
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import MultiLineString
    >>> line_merge(MultiLineString([[(0, 2), (0, 10)], [(0, 10), (5, 10)]]))
    <LINESTRING (0 2, 0 10, 5 10)>
    >>> line_merge(MultiLineString([[(0, 2), (0, 10)], [(0, 11), (5, 10)]]))
    <MULTILINESTRING ((0 2, 0 10), (0 11, 5 10))>
    >>> line_merge(MultiLineString())
    <GEOMETRYCOLLECTION EMPTY>
    >>> line_merge(MultiLineString([[(0, 0), (1, 0)], [(0, 0), (3, 0)]]))
    <LINESTRING (1 0, 0 0, 3 0)>
    >>> line_merge(MultiLineString([[(0, 0), (1, 0)], [(0, 0), (3, 0)]]), directed=True)
    <MULTILINESTRING ((0 0, 1 0), (0 0, 3 0))>
    """
    if directed:
        if lib.geos_version < (3, 11, 0):
            raise UnsupportedGEOSVersionError(
                "'{}' requires at least GEOS {}.{}.{}.".format(
                    "line_merge", *(3, 11, 0)
                )
            )
        return lib.line_merge_directed(line, **kwargs)
    return lib.line_merge(line, **kwargs)


@multithreading_enabled
def shared_paths(
    a: MaybeArrayNLike["LineString"], b: MaybeArrayNLike["LineString"], **kwargs
) -> MaybeArrayN["GeometryCollection"]:
    """Returns the shared paths between the two given LineString geometries.

    Parameters
    ----------
    a, b : MaybeArrayN["LineString"]
        Both geometries should be linestrings or arrays of LineStrings.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    A geometrycollection or array of geometrycollections is returned
    with two elements in each geometrycollection:
    - The first element is a MultiLineString containing shared paths
      with the same direction for both inputs.
    - The second element is a multilinestring containing shared paths
      with the opposite direction for the two inputs.

    Examples
    --------
    >>> from shapely import LineString
    >>> line1 = LineString([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> line2 = LineString([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)])
    >>> shared_paths(line1, line2).wkt
    'GEOMETRYCOLLECTION (MULTILINESTRING EMPTY, MULTILINESTRING ((1 0, 1 1)))'
    >>> line3 = LineString([(1, 1), (0, 1)])
    >>> shared_paths(line1, line3).wkt
    'GEOMETRYCOLLECTION (MULTILINESTRING ((1 1, 0 1)), MULTILINESTRING EMPTY)'
    """
    return lib.shared_paths(a, b, **kwargs)


@multithreading_enabled
def shortest_line(
    a: MaybeGeometryArrayNLike, b: MaybeGeometryArrayNLike, **kwargs
) -> MaybeArrayN["LineString"]:
    """
    Returns the shortest line between two geometries.

    The resulting line consists of two points, representing the nearest
    points between the geometry pair. The line always starts in the first
    geometry `a` and ends in the second geometry `b`. The endpoints of the
    line will not necessarily be existing vertices of the input geometries
    `a` and `b`, but can also be a point along a line segment.

    Parameters
    ----------
    a, b : MaybeGeometryArrayNLike
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    prepare : improve performance by preparing ``a`` (the first argument) (for GEOS>=3.9)

    Examples
    --------
    >>> from shapely import LineString
    >>> line1 = LineString([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> line2 = LineString([(0, 3), (3, 0), (5, 3)])
    >>> shortest_line(line1, line2)
    <LINESTRING (1 1, 1.5 1.5)>
    """
    return lib.shortest_line(a, b, **kwargs)
