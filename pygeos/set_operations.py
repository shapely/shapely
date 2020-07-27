import numpy as np
from . import lib, Geometry, GeometryType
from .decorators import requires_geos
from .decorators import multithreading_enabled

__all__ = [
    "difference",
    "intersection",
    "intersection_all",
    "symmetric_difference",
    "symmetric_difference_all",
    "union",
    "union_all",
    "coverage_union",
    "coverage_union_all",
]

@multithreading_enabled
def difference(a, b, **kwargs):
    """Returns the part of geometry A that does not intersect with geometry B.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like

    Examples
    --------
    >>> line = Geometry("LINESTRING (0 0, 2 2)")
    >>> difference(line, Geometry("LINESTRING (1 1, 3 3)"))
    <pygeos.Geometry LINESTRING (0 0, 1 1)>
    >>> difference(line, Geometry("LINESTRING EMPTY"))
    <pygeos.Geometry LINESTRING (0 0, 2 2)>
    >>> difference(line, None) is None
    True
    """
    return lib.difference(a, b, **kwargs)

@multithreading_enabled
def intersection(a, b, **kwargs):
    """Returns the geometry that is shared between input geometries.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like

    See also
    --------
    intersection_all

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 2 2)")
    >>> intersection(line, Geometry("LINESTRING(1 1, 3 3)"))
    <pygeos.Geometry LINESTRING (1 1, 2 2)>
    """
    return lib.intersection(a, b, **kwargs)

@multithreading_enabled
def intersection_all(geometries, axis=0, **kwargs):
    """Returns the intersection of multiple geometries.

    Parameters
    ----------
    geometries : array_like
    axis : int
        Axis along which the operation is performed. The default (zero)
        performs the operation over the first dimension of the input array.
        axis may be negative, in which case it counts from the last to the
        first axis.

    See also
    --------
    intersection

    Examples
    --------
    >>> line_1 = Geometry("LINESTRING(0 0, 2 2)")
    >>> line_2 = Geometry("LINESTRING(1 1, 3 3)")
    >>> intersection_all([line_1, line_2])
    <pygeos.Geometry LINESTRING (1 1, 2 2)>
    >>> intersection_all([[line_1, line_2, None]], axis=1).tolist()
    [None]
    """
    return lib.intersection.reduce(geometries, axis=axis, **kwargs)

@multithreading_enabled
def symmetric_difference(a, b, **kwargs):
    """Returns the geometry that represents the portions of input geometries
    that do not intersect.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like

    See also
    --------
    symmetric_difference_all

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 2 2)")
    >>> symmetric_difference(line, Geometry("LINESTRING(1 1, 3 3)"))
    <pygeos.Geometry MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>
    """
    return lib.symmetric_difference(a, b, **kwargs)

@multithreading_enabled
def symmetric_difference_all(geometries, axis=0, **kwargs):
    """Returns the symmetric difference of multiple geometries.

    Parameters
    ----------
    geometries : array_like
    axis : int
        Axis along which the operation is performed. The default (zero)
        performs the operation over the first dimension of the input array.
        axis may be negative, in which case it counts from the last to the
        first axis.

    See also
    --------
    symmetric_difference

    Examples
    --------
    >>> line_1 = Geometry("LINESTRING(0 0, 2 2)")
    >>> line_2 = Geometry("LINESTRING(1 1, 3 3)")
    >>> symmetric_difference_all([line_1, line_2])
    <pygeos.Geometry MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))>
    >>> symmetric_difference_all([[line_1, line_2, None]], axis=1).tolist()
    [None]
    """
    return lib.symmetric_difference.reduce(geometries, axis=axis, **kwargs)

@multithreading_enabled
def union(a, b, **kwargs):
    """Merges geometries into one.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like

    See also
    --------
    union_all

    Examples
    --------
    >>> line = Geometry("LINESTRING(0 0, 2 2)")
    >>> union(line, Geometry("LINESTRING(2 2, 3 3)"))
    <pygeos.Geometry MULTILINESTRING ((0 0, 2 2), (2 2, 3 3))>
    >>> union(line, None) is None
    True
    """
    return lib.union(a, b, **kwargs)

@multithreading_enabled
def union_all(geometries, axis=0, **kwargs):
    """Returns the union of multiple geometries.

    Parameters
    ----------
    geometries : array_like
    axis : int
        Axis along which the operation is performed. The default (zero)
        performs the operation over the first dimension of the input array.
        axis may be negative, in which case it counts from the last to the
        first axis.

    See also
    --------
    union

    Examples
    --------
    >>> line_1 = Geometry("LINESTRING(0 0, 2 2)")
    >>> line_2 = Geometry("LINESTRING(2 2, 3 3)")
    >>> union_all([line_1, line_2])
    <pygeos.Geometry MULTILINESTRING ((0 0, 2 2), (2 2, 3 3))>
    >>> union_all([[line_1, line_2, None]], axis=1).tolist()
    [<pygeos.Geometry MULTILINESTRING ((0 0, 2 2), (2 2, 3 3))>]
    """
    # for union_all, GEOS provides an efficient route through first creating
    # GeometryCollections
    # first roll the aggregation axis backwards
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(
            np.asarray(geometries), axis=axis, start=geometries.ndim
        )
    # create_collection acts on the inner axis
    collections = lib.create_collection(geometries, GeometryType.GEOMETRYCOLLECTION)
    return lib.unary_union(collections, **kwargs)


@requires_geos("3.8.0")
@multithreading_enabled
def coverage_union(a, b, **kwargs):
    """Merges multiple polygons into one. This is an optimized version of
    union which assumes the polygons to be non-overlapping.

    Requires at least GEOS 3.8.0.

    Parameters
    ----------
    a : Geometry or array_like
    b : Geometry or array_like

    See also
    --------
    coverage_union_all

    Examples
    --------
    >>> from pygeos.constructive import normalize
    >>> polygon = Geometry("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    >>> normalize(coverage_union(polygon, Geometry("POLYGON ((1 0, 1 1, 2 1, 2 0, 1 0))")))
    <pygeos.Geometry POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))>

    Union with None returns same polygon
    >>> normalize(coverage_union(polygon, None))
    <pygeos.Geometry POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>
    """
    return coverage_union_all([a, b], **kwargs)


@requires_geos("3.8.0")
@multithreading_enabled
def coverage_union_all(geometries, axis=0, **kwargs):
    """Returns the union of multiple polygons of a geometry collection.
    This is an optimized version of union which assumes the polygons
    to be non-overlapping.

    Requires at least GEOS 3.8.0.

    Parameters
    ----------
    geometries : array_like
    axis : int
        Axis along which the operation is performed. The default (zero)
        performs the operation over the first dimension of the input array.
        axis may be negative, in which case it counts from the last to the
        first axis.

    See also
    --------
    coverage_union

    Examples
    --------
    >>> from pygeos.constructive import normalize
    >>> polygon_1 = Geometry("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    >>> polygon_2 = Geometry("POLYGON ((1 0, 1 1, 2 1, 2 0, 1 0))")
    >>> normalize(coverage_union_all([polygon_1, polygon_2]))
    <pygeos.Geometry POLYGON ((0 0, 0 1, 1 1, 2 1, 2 0, 1 0, 0 0))>
    """
    # coverage union in GEOS works over GeometryCollections
    # first roll the aggregation axis backwards
    geometries = np.asarray(geometries)
    if axis is None:
        geometries = geometries.ravel()
    else:
        geometries = np.rollaxis(
            np.asarray(geometries), axis=axis, start=geometries.ndim
        )
    # create_collection acts on the inner axis
    collections = lib.create_collection(geometries, GeometryType.GEOMETRYCOLLECTION)
    return lib.coverage_union(collections, **kwargs)
