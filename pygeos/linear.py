from warnings import warn

from . import lib
from . import Geometry  # NOQA
from .decorators import multithreading_enabled

__all__ = ["line_interpolate_point", "line_locate_point", "line_merge", "shared_paths"]


@multithreading_enabled
def line_interpolate_point(line, distance, normalized=False, **kwargs):
    """Returns a point interpolated at given distance on a line.

    Parameters
    ----------
    line : Geometry or array_like
        For multilinestrings or geometrycollections, the first geometry is taken
        and the rest is ignored. This function raises a TypeError for non-linear
        geometries. For empty linear geometries, empty points are returned.
    distance : float or array_like
        Negative values measure distance from the end of the line. Out-of-range
        values will be clipped to the line endings.
    normalized : bool
        If normalized is set to True, the distance is a fraction of the total
        line length instead of the absolute distance.

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 2, 0 10)")
    >>> line_interpolate_point(line, 2)
    <pygeos.Geometry POINT (0 4)>
    >>> line_interpolate_point(line, 100)
    <pygeos.Geometry POINT (0 10)>
    >>> line_interpolate_point(line, -2)
    <pygeos.Geometry POINT (0 8)>
    >>> line_interpolate_point(line, [0.25, -0.25], normalized=True).tolist()
    [<pygeos.Geometry POINT (0 4)>, <pygeos.Geometry POINT (0 8)>]
    >>> line_interpolate_point(Geometry("LINESTRING EMPTY"), 1)
    <pygeos.Geometry POINT EMPTY>
    """
    if "normalize" in kwargs:
        warn("argument 'normalize' is deprecated; use 'normalized'", DeprecationWarning)
        normalized = kwargs.pop("normalize")
    if normalized:
        return lib.line_interpolate_point_normalized(line, distance)
    else:
        return lib.line_interpolate_point(line, distance)


@multithreading_enabled
def line_locate_point(line, other, normalized=False, **kwargs):
    """Returns the distance to the line origin of given point.

    If given point does not intersect with the line, the point will first be
    projected onto the line after which the distance is taken.

    Parameters
    ----------
    line : Geometry or array_like
    point : Geometry or array_like
    normalized : bool
        If normalized is set to True, the distance is a fraction of the total
        line length instead of the absolute distance.

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 2, 0 10)")
    >>> line_locate_point(line, Geometry("POINT(4 4)"))
    2.0
    >>> line_locate_point(line, Geometry("POINT(4 4)"), normalized=True)
    0.25
    >>> line_locate_point(line, Geometry("POINT(0 18)"))
    8.0
    >>> line_locate_point(Geometry("LINESTRING EMPTY"), Geometry("POINT(4 4)"))
    nan
    """
    if "normalize" in kwargs:
        warn("argument 'normalize' is deprecated; use 'normalized'", DeprecationWarning)
        normalized = kwargs.pop("normalize")
    if normalized:
        return lib.line_locate_point_normalized(line, other)
    else:
        return lib.line_locate_point(line, other)


@multithreading_enabled
def line_merge(line):
    """Returns (multi)linestrings formed by combining the lines in a
    multilinestrings.

    Parameters
    ----------
    line : Geometry or array_like

    Examples
    --------
    >>> line_merge(Geometry("MULTILINESTRING((0 2, 0 10), (0 10, 5 10))"))
    <pygeos.Geometry LINESTRING (0 2, 0 10, 5 10)>
    >>> line_merge(Geometry("MULTILINESTRING((0 2, 0 10), (0 11, 5 10))"))
    <pygeos.Geometry MULTILINESTRING ((0 2, 0 10), (0 11, 5 10))>
    >>> line_merge(Geometry("LINESTRING EMPTY"))
    <pygeos.Geometry GEOMETRYCOLLECTION EMPTY>
    """
    return lib.line_merge(line)


def shared_paths(a, b, **kwargs):
    """Returns the shared paths between geom1 and geom2.

    Both geometries should be linestrings or arrays of linestrings.
    A geometrycollection or array of geometrycollections is returned
    with two elements in each geometrycollection. The first element is a
    multilinestring containing shared paths with the same direction
    for both inputs. The second element is a multilinestring containing
    shared paths with the opposite direction for the two inputs.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like

    Examples
    --------
    >>> geom1 = Geometry("LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)")
    >>> geom2 = Geometry("LINESTRING (1 0, 2 0, 2 1, 1 1, 1 0)")
    >>> shared_paths(geom1, geom2)
    <pygeos.Geometry GEOMETRYCOLLECTION (MULTILINESTRING EMPTY, MULTILINESTRING ...>
    """
    return lib.shared_paths(a, b, **kwargs)
