from enum import IntEnum

import numpy as np

from . import Geometry  # NOQA
from . import _geometry, lib
from .decorators import multithreading_enabled, requires_geos

__all__ = [
    "GeometryType",
    "get_type_id",
    "get_dimensions",
    "get_coordinate_dimension",
    "get_num_coordinates",
    "get_srid",
    "set_srid",
    "get_x",
    "get_y",
    "get_z",
    "get_exterior_ring",
    "get_num_points",
    "get_num_interior_rings",
    "get_num_geometries",
    "get_point",
    "get_interior_ring",
    "get_geometry",
    "get_parts",
    "get_rings",
    "get_precision",
    "set_precision",
]


class GeometryType(IntEnum):
    """The enumeration of GEOS geometry types"""

    MISSING = -1
    POINT = 0
    LINESTRING = 1
    LINEARRING = 2
    POLYGON = 3
    MULTIPOINT = 4
    MULTILINESTRING = 5
    MULTIPOLYGON = 6
    GEOMETRYCOLLECTION = 7


# generic


@multithreading_enabled
def get_type_id(geometry, **kwargs):
    """Returns the type ID of a geometry.

    - None (missing) is -1
    - POINT is 0
    - LINESTRING is 1
    - LINEARRING is 2
    - POLYGON is 3
    - MULTIPOINT is 4
    - MULTILINESTRING is 5
    - MULTIPOLYGON is 6
    - GEOMETRYCOLLECTION is 7

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    GeometryType

    Examples
    --------
    >>> get_type_id(Geometry("LINESTRING (0 0, 1 1, 2 2, 3 3)"))
    1
    >>> get_type_id([Geometry("POINT (1 2)"), Geometry("POINT (1 2)")]).tolist()
    [0, 0]
    """
    return lib.get_type_id(geometry, **kwargs)


@multithreading_enabled
def get_dimensions(geometry, **kwargs):
    """Returns the inherent dimensionality of a geometry.

    The inherent dimension is 0 for points, 1 for linestrings and linearrings,
    and 2 for polygons. For geometrycollections it is the max of the containing
    elements. Empty and None geometries return -1.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> get_dimensions(Geometry("POINT (0 0)"))
    0
    >>> get_dimensions(Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))"))
    2
    >>> get_dimensions(Geometry("GEOMETRYCOLLECTION (POINT(0 0), LINESTRING(0 0, 1 1))"))
    1
    >>> get_dimensions(Geometry("GEOMETRYCOLLECTION EMPTY"))
    -1
    >>> get_dimensions(None)
    -1
    """
    return lib.get_dimensions(geometry, **kwargs)


@multithreading_enabled
def get_coordinate_dimension(geometry, **kwargs):
    """Returns the dimensionality of the coordinates in a geometry (2 or 3).

    Returns -1 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> get_coordinate_dimension(Geometry("POINT (0 0)"))
    2
    >>> get_coordinate_dimension(Geometry("POINT Z (0 0 0)"))
    3
    >>> get_coordinate_dimension(None)
    -1
    """
    return lib.get_coordinate_dimension(geometry, **kwargs)


@multithreading_enabled
def get_num_coordinates(geometry, **kwargs):
    """Returns the total number of coordinates in a geometry.

    Returns 0 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> get_num_coordinates(Geometry("POINT (0 0)"))
    1
    >>> get_num_coordinates(Geometry("POINT Z (0 0 0)"))
    1
    >>> get_num_coordinates(Geometry("GEOMETRYCOLLECTION (POINT(0 0), LINESTRING(0 0, 1 1))"))
    3
    >>> get_num_coordinates(None)
    0
    """
    return lib.get_num_coordinates(geometry, **kwargs)


@multithreading_enabled
def get_srid(geometry, **kwargs):
    """Returns the SRID of a geometry.

    Returns -1 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    set_srid

    Examples
    --------
    >>> point = Geometry("POINT (0 0)")
    >>> with_srid = set_srid(point, 4326)
    >>> get_srid(point)
    0
    >>> get_srid(with_srid)
    4326
    """
    return lib.get_srid(geometry, **kwargs)


@multithreading_enabled
def set_srid(geometry, srid, **kwargs):
    """Returns a geometry with its SRID set.

    Parameters
    ----------
    geometry : Geometry or array_like
    srid : int
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    get_srid

    Examples
    --------
    >>> point = Geometry("POINT (0 0)")
    >>> with_srid = set_srid(point, 4326)
    >>> get_srid(point)
    0
    >>> get_srid(with_srid)
    4326
    """
    return lib.set_srid(geometry, np.intc(srid), **kwargs)


# points


@multithreading_enabled
def get_x(point, **kwargs):
    """Returns the x-coordinate of a point

    Parameters
    ----------
    point : Geometry or array_like
        Non-point geometries will result in NaN being returned.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    get_y, get_z

    Examples
    --------
    >>> get_x(Geometry("POINT (1 2)"))
    1.0
    >>> get_x(Geometry("MULTIPOINT (1 1, 1 2)"))
    nan
    """
    return lib.get_x(point, **kwargs)


@multithreading_enabled
def get_y(point, **kwargs):
    """Returns the y-coordinate of a point

    Parameters
    ----------
    point : Geometry or array_like
        Non-point geometries will result in NaN being returned.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    get_x, get_z

    Examples
    --------
    >>> get_y(Geometry("POINT (1 2)"))
    2.0
    >>> get_y(Geometry("MULTIPOINT (1 1, 1 2)"))
    nan
    """
    return lib.get_y(point, **kwargs)


@requires_geos("3.7.0")
@multithreading_enabled
def get_z(point, **kwargs):
    """Returns the z-coordinate of a point.

    Parameters
    ----------
    point : Geometry or array_like
        Non-point geometries or geometries without 3rd dimension will result
        in NaN being returned.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    get_x, get_y

    Examples
    --------
    >>> get_z(Geometry("POINT Z (1 2 3)"))
    3.0
    >>> get_z(Geometry("POINT (1 2)"))
    nan
    >>> get_z(Geometry("MULTIPOINT Z (1 1 1, 2 2 2)"))
    nan
    """
    return lib.get_z(point, **kwargs)


# linestrings


@multithreading_enabled
def get_point(geometry, index, **kwargs):
    """Returns the nth point of a linestring or linearring.

    Parameters
    ----------
    geometry : Geometry or array_like
    index : int or array_like
        Negative values count from the end of the linestring backwards.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    get_num_points

    Examples
    --------
    >>> line = Geometry("LINESTRING (0 0, 1 1, 2 2, 3 3)")
    >>> get_point(line, 1)
    <pygeos.Geometry POINT (1 1)>
    >>> get_point(line, -2)
    <pygeos.Geometry POINT (2 2)>
    >>> get_point(line, [0, 3]).tolist()
    [<pygeos.Geometry POINT (0 0)>, <pygeos.Geometry POINT (3 3)>]
    >>> get_point(Geometry("LINEARRING (0 0, 1 1, 2 2, 0 0)"), 1)
    <pygeos.Geometry POINT (1 1)>
    >>> get_point(Geometry("MULTIPOINT (0 0, 1 1, 2 2, 3 3)"), 1) is None
    True
    >>> get_point(Geometry("POINT (1 1)"), 0) is None
    True
    """
    return lib.get_point(geometry, np.intc(index), **kwargs)


@multithreading_enabled
def get_num_points(geometry, **kwargs):
    """Returns number of points in a linestring or linearring.

    Returns 0 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
        The number of points in geometries other than linestring or linearring
        equals zero.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    get_point
    get_num_geometries

    Examples
    --------
    >>> line = Geometry("LINESTRING (0 0, 1 1, 2 2, 3 3)")
    >>> get_num_points(line)
    4
    >>> get_num_points(Geometry("MULTIPOINT (0 0, 1 1, 2 2, 3 3)"))
    0
    >>> get_num_points(None)
    0
    """
    return lib.get_num_points(geometry, **kwargs)


# polygons


@multithreading_enabled
def get_exterior_ring(geometry, **kwargs):
    """Returns the exterior ring of a polygon.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    get_interior_ring

    Examples
    --------
    >>> get_exterior_ring(Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))"))
    <pygeos.Geometry LINEARRING (0 0, 0 10, 10 10, 10 0, 0 0)>
    >>> get_exterior_ring(Geometry("POINT (1 1)")) is None
    True
    """
    return lib.get_exterior_ring(geometry, **kwargs)


@multithreading_enabled
def get_interior_ring(geometry, index, **kwargs):
    """Returns the nth interior ring of a polygon.

    Parameters
    ----------
    geometry : Geometry or array_like
    index : int or array_like
        Negative values count from the end of the interior rings backwards.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    get_exterior_ring
    get_num_interior_rings

    Examples
    --------
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))")
    >>> get_interior_ring(polygon_with_hole, 0)
    <pygeos.Geometry LINEARRING (2 2, 2 4, 4 4, 4 2, 2 2)>
    >>> get_interior_ring(Geometry("POINT (1 1)"), 0) is None
    True
    """
    return lib.get_interior_ring(geometry, np.intc(index), **kwargs)


@multithreading_enabled
def get_num_interior_rings(geometry, **kwargs):
    """Returns number of internal rings in a polygon

    Returns 0 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
        The number of interior rings in non-polygons equals zero.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    get_exterior_ring
    get_interior_ring

    Examples
    --------
    >>> polygon = Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))")
    >>> get_num_interior_rings(polygon)
    0
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))")
    >>> get_num_interior_rings(polygon_with_hole)
    1
    >>> get_num_interior_rings(Geometry("POINT (1 1)"))
    0
    >>> get_num_interior_rings(None)
    0
    """
    return lib.get_num_interior_rings(geometry, **kwargs)


# collections


@multithreading_enabled
def get_geometry(geometry, index, **kwargs):
    """Returns the nth geometry from a collection of geometries.

    Parameters
    ----------
    geometry : Geometry or array_like
    index : int or array_like
        Negative values count from the end of the collection backwards.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Notes
    -----
    - simple geometries act as length-1 collections
    - out-of-range values return None

    See also
    --------
    get_num_geometries, get_parts

    Examples
    --------
    >>> multipoint = Geometry("MULTIPOINT (0 0, 1 1, 2 2, 3 3)")
    >>> get_geometry(multipoint, 1)
    <pygeos.Geometry POINT (1 1)>
    >>> get_geometry(multipoint, -1)
    <pygeos.Geometry POINT (3 3)>
    >>> get_geometry(multipoint, 5) is None
    True
    >>> get_geometry(Geometry("POINT (1 1)"), 0)
    <pygeos.Geometry POINT (1 1)>
    >>> get_geometry(Geometry("POINT (1 1)"), 1) is None
    True
    """
    return lib.get_geometry(geometry, np.intc(index), **kwargs)


def get_parts(geometry, return_index=False):
    """Gets parts of each GeometryCollection or Multi* geometry object; returns
    a copy of each geometry in the GeometryCollection or Multi* geometry object.

    Note: This does not return the individual parts of Multi* geometry objects in
    a GeometryCollection.  You may need to call this function multiple times to
    return individual parts of Multi* geometry objects in a GeometryCollection.

    Parameters
    ----------
    geometry : Geometry or array_like
    return_index : bool, default False
        If True, will return a tuple of ndarrays of (parts, indexes), where indexes
        are the indexes of the original geometries in the source array.

    Returns
    -------
    ndarray of parts or tuple of (parts, indexes)

    See also
    --------
    get_geometry, get_rings

    Examples
    --------
    >>> get_parts(Geometry("MULTIPOINT (0 1, 2 3)")).tolist()
    [<pygeos.Geometry POINT (0 1)>, <pygeos.Geometry POINT (2 3)>]
    >>> parts, index = get_parts([Geometry("MULTIPOINT (0 1)"), Geometry("MULTIPOINT (4 5, 6 7)")], return_index=True)
    >>> parts.tolist()
    [<pygeos.Geometry POINT (0 1)>, <pygeos.Geometry POINT (4 5)>, <pygeos.Geometry POINT (6 7)>]
    >>> index.tolist()
    [0, 1, 1]
    """
    geometry = np.asarray(geometry, dtype=np.object_)
    geometry = np.atleast_1d(geometry)

    if geometry.ndim != 1:
        raise ValueError("Array should be one dimensional")

    if return_index:
        return _geometry.get_parts(geometry)

    return _geometry.get_parts(geometry)[0]


def get_rings(geometry, return_index=False):
    """Gets rings of Polygon geometry object.

    For each Polygon, the first returned ring is always the exterior ring
    and potential subsequent rings are interior rings.

    If the geometry is not a Polygon, nothing is returned (empty array for
    scalar geometry input or no element in output array for array input).

    Parameters
    ----------
    geometry : Geometry or array_like
    return_index : bool, default False
        If True, will return a tuple of ndarrays of (rings, indexes), where
        indexes are the indexes of the original geometries in the source array.

    Returns
    -------
    ndarray of rings or tuple of (rings, indexes)

    See also
    --------
    get_exterior_ring, get_interior_ring, get_parts

    Examples
    --------
    >>> polygon_with_hole = Geometry("POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), \
(2 2, 2 4, 4 4, 4 2, 2 2))")
    >>> get_rings(polygon_with_hole).tolist()
    [<pygeos.Geometry LINEARRING (0 0, 0 10, 10 10, 10 0, 0 0)>,
     <pygeos.Geometry LINEARRING (2 2, 2 4, 4 4, 4 2, 2 2)>]

    With ``return_index=True``:

    >>> polygon = Geometry("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))")
    >>> rings, index = get_rings([polygon, polygon_with_hole], return_index=True)
    >>> rings.tolist()
    [<pygeos.Geometry LINEARRING (0 0, 2 0, 2 2, 0 2, 0 0)>,
     <pygeos.Geometry LINEARRING (0 0, 0 10, 10 10, 10 0, 0 0)>,
     <pygeos.Geometry LINEARRING (2 2, 2 4, 4 4, 4 2, 2 2)>]
    >>> index.tolist()
    [0, 1, 1]
    """
    geometry = np.asarray(geometry, dtype=np.object_)
    geometry = np.atleast_1d(geometry)

    if geometry.ndim != 1:
        raise ValueError("Array should be one dimensional")

    if return_index:
        return _geometry.get_parts(geometry, extract_rings=True)

    return _geometry.get_parts(geometry, extract_rings=True)[0]


@multithreading_enabled
def get_num_geometries(geometry, **kwargs):
    """Returns number of geometries in a collection.

    Returns 0 for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
        The number of geometries in points, linestrings, linearrings and
        polygons equals one.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    get_num_points
    get_geometry

    Examples
    --------
    >>> get_num_geometries(Geometry("MULTIPOINT (0 0, 1 1, 2 2, 3 3)"))
    4
    >>> get_num_geometries(Geometry("POINT (1 1)"))
    1
    >>> get_num_geometries(None)
    0
    """
    return lib.get_num_geometries(geometry, **kwargs)


@requires_geos("3.6.0")
@multithreading_enabled
def get_precision(geometry, **kwargs):
    """Get the precision of a geometry.

    If a precision has not been previously set, it will be 0 (double
    precision). Otherwise, it will return the precision grid size that was
    set on a geometry.

    Returns NaN for not-a-geometry values.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    set_precision

    Examples
    --------
    >>> get_precision(Geometry("POINT (1 1)"))
    0.0
    >>> geometry = set_precision(Geometry("POINT (1 1)"), 1.0)
    >>> get_precision(geometry)
    1.0
    >>> np.isnan(get_precision(None))
    True
    """
    return lib.get_precision(geometry, **kwargs)


@requires_geos("3.6.0")
@multithreading_enabled
def set_precision(geometry, grid_size, preserve_topology=False, **kwargs):
    """Returns geometry with the precision set to a precision grid size.

    By default, geometries use double precision coordinates (grid_size = 0).

    Coordinates will be rounded if a precision grid is less precise than the
    input geometry. Duplicated vertices will be dropped from lines and
    polygons for grid sizes greater than 0. Line and polygon geometries may
    collapse to empty geometries if all vertices are closer together than
    grid_size. Z values, if present, will not be modified.

    Note: subsequent operations will always be performed in the precision of
    the geometry with higher precision (smaller "grid_size"). That same
    precision will be attached to the operation outputs.

    Also note: input geometries should be geometrically valid; unexpected
    results may occur if input geometries are not.

    Returns None if geometry is None.

    Parameters
    ----------
    geometry : Geometry or array_like
    grid_size : float
        Precision grid size. If 0, will use double precision (will not modify
        geometry if precision grid size was not previously set). If this
        value is more precise than input geometry, the input geometry will
        not be modified.
    preserve_topology : bool, default False
        If True, will attempt to preserve the topology of a geometry after
        rounding coordinates.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    get_precision

    Examples
    --------
    >>> set_precision(Geometry("POINT (0.9 0.9)"), 1.0)
    <pygeos.Geometry POINT (1 1)>
    >>> set_precision(Geometry("POINT (0.9 0.9 0.9)"), 1.0)
    <pygeos.Geometry POINT Z (1 1 0.9)>
    >>> set_precision(Geometry("LINESTRING (0 0, 0 0.1, 0 1, 1 1)"), 1.0)
    <pygeos.Geometry LINESTRING (0 0, 0 1, 1 1)>
    >>> set_precision(None, 1.0) is None
    True
    """

    return lib.set_precision(geometry, grid_size, preserve_topology, **kwargs)
