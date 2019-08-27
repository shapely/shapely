from . import ufuncs
from . import Geometry, Empty  # NOQA

__all__ = ["area", "distance", "length", "hausdorff_distance"]


def area(geometry):
    """Computes the area of a (multi)polygon.

    Parameters
    ----------
    geometry : Geometry or array_like

    Examples
    --------
    >>> area(Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))"))
    100.0
    >>> area(Geometry("MULTIPOLYGON (((0 0, 0 10, 10 10, 0 0)), ((0 0, 0 10, 10 10, 0 0)))"))
    100.0
    >>> area(Empty)
    nan
    """
    return ufuncs.area(geometry)


def distance(a, b):
    """Computes the Cartesian distance between two geometries.

    Parameters
    ----------
    a, b : Geometry or array_like

    Examples
    --------
    >>> point = Geometry("POINT (0 0)")
    >>> distance(Geometry("POINT (10 0)"), point)
    10.0
    >>> distance(Geometry("LINESTRING (1 1, 1 -1)"), point)
    1.0
    >>> distance(Geometry("POLYGON ((3 0, 5 0, 5 5, 3 5, 3 0))"), point)
    3.0
    >>> distance(Empty, point)
    nan
    """
    return ufuncs.distance(a, b)


def length(geometry):
    """Computes the length of a (multi)linestring or polygon perimeter.

    Parameters
    ----------
    geometry : Geometry or array_like

    Examples
    --------
    >>> length(Geometry("LINESTRING (0 0, 0 2, 3 2)"))
    5.0
    >>> length(Geometry("MULTILINESTRING ((0 0, 1 0), (0 0, 1 0))"))
    2.0
    >>> length(Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))"))
    40.0
    >>> length(Empty)
    nan
    """
    return ufuncs.length(geometry)


def hausdorff_distance(a, b, densify=None):
    """Compute the discrete Haussdorf distance between two geometries.

    The Haussdorf distance is a measure of similarity: it is the greatest
    distance between any point in A and the closest point in B. The discrete
    distance is an approximation of this metric: only vertices are considered.
    The parameter 'densify' makes this approximation less coarse by splitting
    the line segments between vertices before computing the distance.

    Parameters
    ----------
    a, b : Geometry or array_like
    densify : float, array_like or None
        The value of densify is required to be between 0 and 1.

    Examples
    --------
    >>> line_1 = Geometry("LINESTRING (130 0, 0 0, 0 150)")
    >>> line_2 = Geometry("LINESTRING (10 10, 10 150, 130 10)")
    >>> hausdorff_distance(line_1, line_2)  # doctest: +ELLIPSIS
    14.14...
    >>> hausdorff_distance(line_1, line_2, densify=0.5)
    70.0
    >>> hausdorff_distance(line_1, Empty)
    nan
    """
    if densify is None:
        return ufuncs.hausdorff_distance(a, b)
    else:
        return ufuncs.haussdorf_distance_densify(a, b, densify)
