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
    "get_point_n",
    "get_geometry_n",
    "get_interior_ring_n",
    "set_srid",
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

    Attributes
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

    Attributes
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

    Attributes
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

    Attributes
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

    Attributes
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

    Attributes
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

    Attributes
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

    Attributes
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

    Attributes
    ----------
    geometries : array_like
        An array of geometries
    """
    return ufuncs.create_collection(geometries, GeometryType.GEOMETRYCOLLECTION)


@wraps(ufuncs.get_geometry_n)
def get_geometry_n(geoms, index):
    return ufuncs.get_geometry_n(geoms, np.intc(index))


@wraps(ufuncs.get_interior_ring_n)
def get_interior_ring_n(geoms, index):
    return ufuncs.get_interior_ring_n(geoms, np.intc(index))


@wraps(ufuncs.get_point_n)
def get_point_n(geoms, index):
    return ufuncs.get_point_n(geoms, np.intc(index))


@wraps(ufuncs.set_srid)
def set_srid(geoms, srid):
    return ufuncs.set_srid(geoms, np.intc(srid))
