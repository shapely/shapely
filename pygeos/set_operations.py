import numpy as np
from . import ufuncs, Empty, Geometry

__all__ = ["difference", "intersection", "symmetric_difference", "union"]


def difference(a, b=None, **kwargs):
    """Returns the part of geometry A that does not intersect with geometry B.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like or None
    axis : int
        The axis in a to perform the difference on. This a.shape[axis] is
        required to be 2. The axis is ignored when b is not None.

    Examples
    --------
    >>> line_1 = Geometry("LINESTRING(0 0, 2 2)")
    >>> line_2 = Geometry("LINESTRING(1 1, 3 3)")
    >>> difference(line_1, line_2)
    <pygeos.Geometry LINESTRING (0 0, 1 1)>
    >>> difference([line_2, line_1])
    <pygeos.Geometry LINESTRING (2 2, 3 3)>
    >>> difference([[line_1, line_2], [line_1, line_1]], axis=1).tolist()
    [<pygeos.Geometry LINESTRING (0 0, 1 1)>, <pygeos.Empty>]
    >>> difference([[line_1, line_2], [line_1, line_1]], axis=0).tolist()
    [<pygeos.Empty>, <pygeos.Geometry LINESTRING (2 2, 3 3)>]
    >>> difference(Empty, line_1)
    <pygeos.Empty>
    >>> difference(line_1, Empty)
    <pygeos.Geometry LINESTRING (0 0, 2 2)>
    """
    if b is None:
        axis = kwargs.get("axis", 0)
        a = np.asarray(a)
        if a.shape[axis] != 2:
            raise ValueError(
                "Can only take a difference over an axis with length 2"
            )
        return ufuncs.difference.reduce(a, **kwargs)
    return ufuncs.difference(a, b, **kwargs)


def intersection(a, b=None, axis=0):
    """Returns the geometry that is shared between input geometries.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like or None
    axis : int
        The axis in a to perform the operation on. This axis can be of any
        shape: all geometries on that axis are intersected. Ignored when b is
        not None.

    Examples
    --------
    >>> line_1 = Geometry("LINESTRING(0 0, 2 2)")
    >>> line_2 = Geometry("LINESTRING(1 1, 3 3)")
    >>> intersection(line_1, line_2)
    <pygeos.Geometry LINESTRING (1 1, 2 2)>
    >>> intersection([line_1, line_2, line_1], axis=0)
    <pygeos.Geometry LINESTRING (1 1, 2 2)>
    >>> intersection([[line_1, Empty]], axis=1).tolist()
    [<pygeos.Empty>]
    >>> intersection(Empty, line_1)
    <pygeos.Empty>
    """
    if b is None:
        return ufuncs.intersection.reduce(a, axis=axis)
    else:
        return ufuncs.intersection(a, b)


def symmetric_difference(a, b=None, axis=0):
    """Returns the geometry that represents the portions of input geometries
    that do not intersect.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like or None
    axis : int
        The axis in a to perform the operation on. This axis can be of any
        shape: all geometries on that axis are intersected. Ignored when b is
        not None.

    Examples
    --------
    >>> line_1 = Geometry("LINESTRING(0 0, 2 2)")
    >>> line_2 = Geometry("LINESTRING(1 1, 3 3)")
    >>> symmetric_difference(line_1, line_2)
    <pygeos.Geometry MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>
    >>> symmetric_difference([line_1, line_1])
    <pygeos.Empty>
    >>> symmetric_difference(line_1, Empty)
    <pygeos.Geometry LINESTRING (0 0, 2 2)>
    """
    if b is None:
        return ufuncs.symmetric_difference.reduce(a, axis=axis)
    else:
        return ufuncs.symmetric_difference(a, b)


def union(a, b=None, axis=0):
    """Combines geometries into one.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like or None
    axis : int
        The axis in a to perform the operation on. This axis can be of any
        shape: all geometries on that axis are intersected. Ignored when b is
        not None.

    Examples
    --------
    >>> line_1 = Geometry("LINESTRING(0 0, 2 2)")
    >>> line_2 = Geometry("LINESTRING(1 1, 3 3)")
    >>> union([line_1, line_2])
    <pygeos.Geometry LINESTRING (0 0, 3 3)>
    >>> union([line_1, line_2, line_2])
    <pygeos.Geometry LINESTRING (0 0, 3 3)>
    >>> union(line_1, Empty)
    <pygeos.Empty>
    """
    if b is None:
        a = ufuncs.create_collection(a, axis=axis)
        return ufuncs.unary_union(a)
    else:
        return ufuncs.union(a, b)
