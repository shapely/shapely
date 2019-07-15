from enum import IntEnum
import numpy as np
from functools import wraps
from . import ufuncs
from .ufuncs import Geometry

__all__ = [
    "Geometry",
    "GeometryType",
    "points",
    "linestrings",
    "linearrings",
    "polygons",
    "multipoints",
    "multilinestrings",
    "multipolygons",
    "geometrycollections",
    "box",
    "get_type_id",
    "get_dimensions",
    "get_coordinate_dimensions",
    "get_num_coordinates",
    "normalize",
    "get_srid",
    "set_srid",
    "get_x",
    "get_y",
    "get_num_points",
    "get_start_point",
    "get_end_point",
    "get_point",
    "get_exterior_ring",
    "get_num_interior_rings",
    "get_interior_ring",
    "get_geometry",
]


class GeometryType(IntEnum):
    POINT = 0
    LINESTRING = 1
    LINEARRING = 2
    POLYGON = 3
    MULTIPOINT = 4
    MULTILINESTRING = 5
    MULTIPOLYGON = 6
    GEOMETRYCOLLECTION = 7


def _wrap_construct_ufunc(func, coords, y=None, z=None):
    if y is None:
        return func(coords)
    x = coords
    if z is None:
        coords = np.broadcast_arrays(x, y)
    else:
        coords = np.broadcast_arrays(x, y, z)
    return func(np.stack(coords, axis=-1))


def points(coords, y=None, z=None):
    """Create an array of points.

    Parameters
    ----------
    coords : array_like
        An array of coordinate tuples (2- or 3-dimensional) or, if `y` is
        provided, an array of x coordinates.
    y : array_like
    z : array_like
    """
    return _wrap_construct_ufunc(ufuncs.points, coords, y, z)


def linestrings(coords, y=None, z=None):
    """Create an array of linestrings.

    Parameters
    ----------
    coords : array_like
        An array of lists of coordinate tuples (2- or 3-dimensional) or, if `y`
        is provided, an array of lists of x coordinates
    y : array_like
    z : array_like
    """
    return _wrap_construct_ufunc(ufuncs.linestrings, coords, y, z)


def linearrings(coords, y=None, z=None):
    """Create an array of linearrings.

    If the provided coords do not constitute a closed linestring, the first
    coordinate is duplicated at the end to close the ring.

    Parameters
    ----------
    coords : array_like
        An array of lists of coordinate tuples (2- or 3-dimensional) or, if `y`
        is provided, an array of lists of x coordinates
    y : array_like
    z : array_like
    """
    return _wrap_construct_ufunc(ufuncs.linearrings, coords, y, z)


def polygons(shells, holes=None):
    """Create an array of polygons.

    Parameters
    ----------
    shell : array_like
        An array of linearrings that constitute the out shell of the polygons.
        Coordinates can also be passed, see linearrings.
    holes : array_like
        An array of lists of linearrings that constitute holes for each shell.
    """
    shells = np.asarray(shells)
    if not isinstance(shells, Geometry) and np.issubdtype(shells.dtype, np.number):
        shells = linearrings(shells)

    if holes is None:
        return ufuncs.polygons_without_holes(shells)

    holes = np.asarray(holes)
    if not isinstance(holes, Geometry) and np.issubdtype(holes.dtype, np.number):
        holes = linearrings(holes)
    return ufuncs.polygons_with_holes(shells, holes)


def box(x1, y1, x2, y2):
    """Create box polygons.

    Parameters
    ----------
    x1 : array_like
    y2 : array_like
    x1 : array_like
    y2 : array_like
    """
    x1, y1, x2, y2 = np.broadcast_arrays(x1, y1, x2, y2)
    rings = np.array(((x2, y1), (x2, y2), (x1, y2), (x1, y1)))
    # bring first two axes to the last two positions
    rings = rings.transpose(list(range(2, rings.ndim)) + [0, 1])
    return polygons(rings)


def multipoints(geometries):
    """Create multipoints from arrays of points

    Parameters
    ----------
    geometries : array_like
        An array of points or coordinates (see points).
    """
    geometries = np.asarray(geometries)
    if not isinstance(geometries, Geometry) and np.issubdtype(
        geometries.dtype, np.number
    ):
        geometries = points(geometries)
    return ufuncs.create_collection(geometries, GeometryType.MULTIPOINT)


def multilinestrings(geometries):
    """Create multilinestrings from arrays of linestrings

    Parameters
    ----------
    geometries : array_like
        An array of linestrings or coordinates (see linestrings).
    """
    geometries = np.asarray(geometries)
    if not isinstance(geometries, Geometry) and np.issubdtype(
        geometries.dtype, np.number
    ):
        geometries = linestrings(geometries)
    return ufuncs.create_collection(geometries, GeometryType.MULTILINESTRING)


def multipolygons(geometries):
    """Create multipolygons from arrays of polygons

    Parameters
    ----------
    geometries : array_like
        An array of polygons or coordinates (see polygons).
    """
    geometries = np.asarray(geometries)
    if not isinstance(geometries, Geometry) and np.issubdtype(
        geometries.dtype, np.number
    ):
        geometries = polygons(geometries)
    return ufuncs.create_collection(geometries, GeometryType.MULTIPOLYGON)


def geometrycollections(geometries):
    """Create geometrycollections from arrays of geometries

    Parameters
    ----------
    geometries : array_like
        An array of geometries
    """
    return ufuncs.create_collection(geometries, GeometryType.GEOMETRYCOLLECTION)


# generic


def get_type_id(geometries):
    return ufuncs.get_type_id(geometries)


def get_dimensions(geometries):
    return ufuncs.get_dimensions(geometries)


def get_coordinate_dimensions(geometries):
    return ufuncs.get_coordinate_dimensions(geometries)


def get_num_coordinates(geometries):
    return ufuncs.get_num_coordinates(geometries)


def normalize(geometries):
    return ufuncs.normalize(geometries)


def get_srid(geometries):
    return ufuncs.get_srid(geometries)


def set_srid(geometries, srid):
    return ufuncs.set_srid(geometries, np.intc(srid))


# points


def get_x(point):
    return ufuncs.get_x(point)


def get_y(point):
    return ufuncs.get_y(point)


# linestrings


def get_num_points(linestring):
    return ufuncs.get_num_points(linestring)


def get_start_point(linestring):
    return ufuncs.get_start_point(linestring)


def get_end_point(linestring):
    return ufuncs.get_end_point(linestring)


def get_point(linestring, index):
    return ufuncs.get_point(linestring, np.intc(index))


# polygons


def get_exterior_ring(polygon):
    return ufuncs.get_exterior_ring(polygon)


def get_num_interior_rings(polygon):
    return ufuncs.get_num_interior_rings(polygon)


def get_interior_ring(polygon, index):
    return ufuncs.get_interior_ring(polygon, np.intc(index))


# collections


def get_geometry(geometries, index):
    return ufuncs.get_geometry(geometries, np.intc(index))
