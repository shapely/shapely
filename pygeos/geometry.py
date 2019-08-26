from enum import IntEnum
import numpy as np
from . import ufuncs

__all__ = [
    "Geometry",
    "Empty",
    "GeometryType",
    "get_type_id",
    "get_dimensions",
    "get_coordinate_dimensions",
    "get_num_coordinates",
    "has_z",
    "normalize",
    "get_srid",
    "set_srid",
    "get_x",
    "get_y",
    "get_exterior_ring",
    "get_num_points",
    "get_num_interior_rings",
    "get_num_geometries",
    "get_point",
    "get_interior_ring",
    "get_geometry",
]


Geometry = ufuncs.Geometry
Empty = ufuncs.Empty


class GeometryType(IntEnum):
    POINT = 0
    LINESTRING = 1
    LINEARRING = 2
    POLYGON = 3
    MULTIPOINT = 4
    MULTILINESTRING = 5
    MULTIPOLYGON = 6
    GEOMETRYCOLLECTION = 7


# generic


def get_type_id(geometry):
    return ufuncs.get_type_id(geometry)


def get_dimensions(geometry):
    return ufuncs.get_dimensions(geometry)


def get_coordinate_dimensions(geometry):
    return ufuncs.get_coordinate_dimensions(geometry)


def get_num_coordinates(geometry):
    return ufuncs.get_num_coordinates(geometry)


def has_z(geometry, **kwargs):
    """Returns True if a geometry has a Z coordinate.

    Parameters
    ----------
    geometry : Geometry or array_like

    Notes
    -----
    Keyword arguments (``**kwargs``) are passed into the underlying ufunc. To
    use methods such as ``.at``, import the underlying ufunc from
    ``pygeos.ufuncs``. See the
    `NumPy docs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_.

    Examples
    --------
    >>> has_z(Geometry("POINT (0 0)"))
    False
    >>> has_z(Geometry("POINT Z (0 0 0)"))
    True
    """
    return ufuncs.has_z(geometry, **kwargs)


def normalize(geometry):
    return ufuncs.normalize(geometry)


def get_srid(geometry):
    return ufuncs.get_srid(geometry)


def set_srid(geometry, srid):
    return ufuncs.set_srid(geometry, np.intc(srid))


# points


def get_x(point):
    return ufuncs.get_x(point)


def get_y(point):
    return ufuncs.get_y(point)


# polygons


def get_exterior_ring(polygon):
    return ufuncs.get_exterior_ring(polygon)


def get_num_points(geometry):
    """Returns number of points in a linestring or linearring.

    Parameters
    ----------
    geometry : Geometry or array_like
        The number of points in geometries other than linestring or linearring
        equals zero.

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
    """
    return ufuncs.get_num_points(geometry)


def get_num_interior_rings(geometry):
    """Returns number of internal rings in a polygon

    Parameters
    ----------
    geometry : Geometry or array_like
        The number of interior rings in non-polygons equals zero.

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
    """
    return ufuncs.get_num_interior_rings(geometry)


def get_num_geometries(geometry):
    """Returns number of geometries in a collection.

    Parameters
    ----------
    geometry : Geometry or array_like
        The number of geometries in points, linestrings, linearrings and
        polygons equals one.

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
    """
    return ufuncs.get_num_geometries(geometry)


def get_point(geometry, index):
    """Returns the nth point of a linestring or linearring.

    Parameters
    ----------
    geometry : Geometry or array_like
    index : int or array_like
        Negative values count from the end of the linestring backwards.

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
    >>> get_point(Geometry("MULTIPOINT (0 0, 1 1, 2 2, 3 3)"), 1)
    <pygeos.Empty>
    >>> get_point(Geometry("POINT (1 1)"), 0)
    <pygeos.Empty>
    """
    return ufuncs.get_point(geometry, np.intc(index))


def get_interior_ring(geometry, index):
    """Returns the nth interior ring of a polygon.

    Parameters
    ----------
    geometry : Geometry or array_like
    index : int or array_like
        Negative values count from the end of the interior rings backwards.

    See also
    --------
    get_exterior_ring
    get_num_interior_rings

    Examples
    --------
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))")
    >>> get_interior_ring(polygon_with_hole, 0)
    <pygeos.Geometry LINEARRING (2 2, 2 4, 4 4, 4 2, 2 2)>
    >>> get_interior_ring(Geometry("POINT (1 1)"), 0)
    <pygeos.Empty>
    """
    return ufuncs.get_interior_ring(geometry, np.intc(index))


def get_geometry(geometry, index):
    """Returns the nth geometry from a collection of geometries.

    Parameters
    ----------
    geometry : Geometry or array_like
    index : int or array_like
        Negative values count from the end of the collection backwards.

    Notes
    -----
    - simple geometries act as length-1 collections
    - out-of-range values return Empty

    See also
    --------
    get_num_geometries

    Examples
    --------
    >>> multipoint = Geometry("MULTIPOINT (0 0, 1 1, 2 2, 3 3)")
    >>> get_geometry(multipoint, 1)
    <pygeos.Geometry POINT (1 1)>
    >>> get_geometry(multipoint, -1)
    <pygeos.Geometry POINT (3 3)>
    >>> get_geometry(multipoint, 5)
    <pygeos.Empty>
    >>> get_geometry(Geometry("POINT (1 1)"), 0)
    <pygeos.Geometry POINT (1 1)>
    >>> get_geometry(Geometry("POINT (1 1)"), 1)
    <pygeos.Empty>
    """
    return ufuncs.get_geometry(geometry, np.intc(index))
