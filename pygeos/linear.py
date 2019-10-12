from . import lib
from . import Geometry  # NOQA

__all__ = ["line_interpolate_point", "line_locate_point", "line_merge"]


def line_interpolate_point(line, distance, normalize=False):
    """Returns a point interpolated at given distance on a line.

    Parameters
    ----------
    line : Geometry or array_like
    distance : float or array_like
        Negative values measure distance from the end of the line. Out-of-range
        values will be clipped to the line endings.
    normalize : bool
        If normalize is set to True, the distance is a fraction of the total
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
    >>> line_interpolate_point(line, [0.25, -0.25], normalize=True).tolist()
    [<pygeos.Geometry POINT (0 4)>, <pygeos.Geometry POINT (0 8)>]
    >>> line_interpolate_point(Geometry("LINESTRING EMPTY"), 1)
    <pygeos.Geometry POINT EMPTY>
    """
    if normalize:
        return lib.line_interpolate_point_normalized(line, distance)
    else:
        return lib.line_interpolate_point(line, distance)


def line_locate_point(line, other, normalize=False):
    """Returns the distance to the line origin of given point.

    If given point does not intersect with the line, the point will first be
    projected onto the line after which the distance is taken.

    Parameters
    ----------
    line : Geometry or array_like
    point : Geometry or array_like
    normalize : bool
        If normalize is set to True, the distance is a fraction of the total
        line length instead of the absolute distance.

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 2, 0 10)")
    >>> line_locate_point(line, Geometry("POINT(4 4)"))
    2.0
    >>> line_locate_point(line, Geometry("POINT(4 4)"), normalize=True)
    0.25
    >>> line_locate_point(line, Geometry("POINT(0 18)"))
    8.0
    >>> line_locate_point(Geometry("LINESTRING EMPTY"), Geometry("POINT(4 4)"))
    nan
    """
    if normalize:
        return lib.line_locate_point_normalized(line, other)
    else:
        return lib.line_locate_point(line, other)


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
