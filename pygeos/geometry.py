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
    "get_start_point",
    "get_end_point",
    "get_exterior_ring",
    "get_num_elements",
    "get_element",
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


# linestrings


def get_start_point(linestring):
    return ufuncs.get_start_point(linestring)


def get_end_point(linestring):
    return ufuncs.get_end_point(linestring)


# polygons


def get_exterior_ring(polygon):
    return ufuncs.get_exterior_ring(polygon)


# collections


def get_num_elements(collection):
    return ufuncs.get_num_elements(collection)


def get_element(geometry, index):
    """Returns the nth sub-element from a geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
    index : int or array_like

    Notes
    -----
    - points are indexed as length-1 collections
    - elements of linestrings and linearrings are points
    - elements of polygons are the interior linearrings

    See also
    --------
    get_exterior_ring
    get_num_elements

    Examples
    --------
    >>> line = Geometry("LINESTRING (0 0, 1 1, 2 2, 3 3)")
    >>> get_element(line, 1)
    <pygeos.Geometry POINT (1 1)>
    >>> get_element(line, -2)
    <pygeos.Geometry POINT (2 2)>
    >>> get_element(line, [0, 3]).tolist()
    [<pygeos.Geometry POINT (0 0)>, <pygeos.Geometry POINT (3 3)>]
    >>> get_element(Geometry("LINEARRING (0 0, 1 1, 2 2, 0 0)"), 1)
    <pygeos.Geometry POINT (1 1)>
    >>> get_element(Geometry("MULTIPOINT (0 0, 1 1, 2 2, 3 3)"), 1)
    <pygeos.Geometry POINT (1 1)>
    >>> polygon_with_hole = Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))")
    >>> get_element(polygon_with_hole, 0)
    <pygeos.Geometry LINEARRING (2 2, 2 4, 4 4, 4 2, 2 2)>
    >>> get_element(Geometry("POINT (1 1)"), 0)
    <pygeos.Geometry POINT (1 1)>
    """
    return ufuncs.get_element(geometry, np.intc(index))
