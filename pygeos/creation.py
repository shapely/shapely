import numpy as np
from . import lib
from . import Geometry, GeometryType
from .decorators import multithreading_enabled

__all__ = [
    "points",
    "linestrings",
    "linearrings",
    "polygons",
    "multipoints",
    "multilinestrings",
    "multipolygons",
    "geometrycollections",
    "box",
    "prepare",
    "destroy_prepared",
]


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
    return _wrap_construct_ufunc(lib.points, coords, y, z)


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
    return _wrap_construct_ufunc(lib.linestrings, coords, y, z)


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
    return _wrap_construct_ufunc(lib.linearrings, coords, y, z)

@multithreading_enabled
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
        return lib.polygons_without_holes(shells)

    holes = np.asarray(holes)
    if not isinstance(holes, Geometry) and np.issubdtype(holes.dtype, np.number):
        holes = linearrings(holes)
    return lib.polygons_with_holes(shells, holes)


def box(x1, y1, x2, y2):
    """Create box polygons.

    Parameters
    ----------
    x1 : array_like
    y1 : array_like
    x2 : array_like
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
    return lib.create_collection(geometries, GeometryType.MULTIPOINT)


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
    return lib.create_collection(geometries, GeometryType.MULTILINESTRING)


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
    return lib.create_collection(geometries, GeometryType.MULTIPOLYGON)


def geometrycollections(geometries):
    """Create geometrycollections from arrays of geometries

    Parameters
    ----------
    geometries : array_like
        An array of geometries
    """
    return lib.create_collection(geometries, GeometryType.GEOMETRYCOLLECTION)


def prepare(geometry, **kwargs):
    """Prepare a geometry, improving performance of other operations.

    A prepared geometry is a normal geometry with added information such as an
    index on the line segments. This improves the performance of the following operations:
    contains, contains_properly, covered_by, covers, crosses, disjoint, intersects,
    overlaps, touches, and within.

    Note that if a prepared geometry is modified, the newly created Geometry object is
    not prepared. In that case, ``prepare`` should be called again.

    This function does not recompute previously prepared geometries; 
    it is efficient to call this function on an array that partially contains prepared geometries. 

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometries are changed inplace

    See also
    --------
    is_prepared : Identify whether a geometry is prepared already.
    destroy_prepared : Destroy the prepared part of a geometry.
    """
    lib.prepare(geometry, **kwargs)


def destroy_prepared(geometry, **kwargs):
    """Destroy the prepared part of a geometry, freeing up memory.

    Note that the prepared geometry will always be cleaned up if the geometry itself
    is dereferenced. This function needs only be called in very specific circumstances,
    such as freeing up memory without losing the geometries, or benchmarking.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometries are changed inplace

    See also
    --------
    prepare
    """
    lib.destroy_prepared(geometry, **kwargs)
