import numpy as np

from . import Geometry, GeometryType, lib
from ._geometry import collections_1d, simple_geometries_1d
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


def _xyz_to_coords(x, y, z):
    if y is None:
        return x
    if z is None:
        coords = np.broadcast_arrays(x, y)
    else:
        coords = np.broadcast_arrays(x, y, z)
    return np.stack(coords, axis=-1)


@multithreading_enabled
def points(coords, y=None, z=None, indices=None, **kwargs):
    """Create an array of points.

    Note that GEOS >=3.10 automatically converts POINT (nan nan) to
    POINT EMPTY.

    Parameters
    ----------
    coords : array_like
        An array of coordinate tuples (2- or 3-dimensional) or, if `y` is
        provided, an array of x coordinates.
    y : array_like
    z : array_like
    indices : array_like or None
       Indices into the target array where input coordinates belong. If
       provided, the coords should be 2D with shape (N, 2) or (N, 3) and
       indices should be 1D with shape (N,). Missing indices will give None
       values in the output array.
    """
    coords = _xyz_to_coords(coords, y, z)
    if indices is None:
        return lib.points(coords, **kwargs)
    else:
        return simple_geometries_1d(coords, indices, GeometryType.POINT)


@multithreading_enabled
def linestrings(coords, y=None, z=None, indices=None, **kwargs):
    """Create an array of linestrings.

    This function will raise an exception if a linestring contains less than
    two points.

    Parameters
    ----------
    coords : array_like
        An array of lists of coordinate tuples (2- or 3-dimensional) or, if `y`
        is provided, an array of lists of x coordinates
    y : array_like
    z : array_like
    indices : array_like or None
       Indices into the target array where input coordinates belong. If
       provided, the coords should be 2D with shape (N, 2) or (N, 3) and
       indices should be 1D with shape (N,). Missing indices will give None
       values in the output array.
    """
    coords = _xyz_to_coords(coords, y, z)
    if indices is None:
        return lib.linestrings(coords, **kwargs)
    else:
        return simple_geometries_1d(coords, indices, GeometryType.LINESTRING)


@multithreading_enabled
def linearrings(coords, y=None, z=None, indices=None, **kwargs):
    """Create an array of linearrings.

    If the provided coords do not constitute a closed linestring, the first
    coordinate is duplicated at the end to close the ring. This function will
    raise an exception if a linearring contains less than three points or if
    the terminal coordinates contain NaN (not-a-number).

    Parameters
    ----------
    coords : array_like
        An array of lists of coordinate tuples (2- or 3-dimensional) or, if `y`
        is provided, an array of lists of x coordinates
    y : array_like
    z : array_like
    indices : array_like or None
       Indices into the target array where input coordinates belong. If
       provided, the coords should be 2D with shape (N, 2) or (N, 3) and
       indices should be 1D with shape (N,). Missing indices will give None
       values in the output array.
    """
    coords = _xyz_to_coords(coords, y, z)
    if indices is None:
        return lib.linearrings(coords, **kwargs)
    else:
        return simple_geometries_1d(coords, indices, GeometryType.LINEARRING)


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


def box(xmin, ymin, xmax, ymax, ccw=True, **kwargs):
    """Create box polygons.

    Parameters
    ----------
    xmin : array_like
    ymin : array_like
    xmax : array_like
    ymax : array_like
    ccw : bool (default: True)
        If True, box will be created in counterclockwise direction starting
        from bottom right coordinate (xmax, ymin).
        If False, box will be created in clockwise direction starting from
        bottom left coordinate (xmin, ymin).

    Examples
    --------
    >>> box(0, 0, 1, 1)
    <pygeos.Geometry POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))>
    >>> box(0, 0, 1, 1, ccw=False)
    <pygeos.Geometry POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>

    """
    return lib.box(xmin, ymin, xmax, ymax, ccw, **kwargs)


def multipoints(geometries, indices=None):
    """Create multipoints from arrays of points

    Parameters
    ----------
    geometries : array_like
        An array of points or coordinates (see points).
    indices : array_like or None
       Indices into the target array where input geometries belong. If
        provided, both geometries and indices should be 1D and have matching
        sizes.
    """
    typ = GeometryType.MULTIPOINT
    geometries = np.asarray(geometries)
    if not isinstance(geometries, Geometry) and np.issubdtype(
        geometries.dtype, np.number
    ):
        geometries = points(geometries)
    if indices is None:
        return lib.create_collection(geometries, typ)
    else:
        return collections_1d(geometries, indices, typ)


def multilinestrings(geometries, indices=None):
    """Create multilinestrings from arrays of linestrings

    Parameters
    ----------
    geometries : array_like
        An array of linestrings or coordinates (see linestrings).
    indices : array_like or None
        Indices into the target array where input geometries belong. If
        provided, both geometries and indices should be 1D and have matching
        sizes.
    """
    typ = GeometryType.MULTILINESTRING
    geometries = np.asarray(geometries)
    if not isinstance(geometries, Geometry) and np.issubdtype(
        geometries.dtype, np.number
    ):
        geometries = linestrings(geometries)

    if indices is None:
        return lib.create_collection(geometries, typ)
    else:
        return collections_1d(geometries, indices, typ)


def multipolygons(geometries, indices=None):
    """Create multipolygons from arrays of polygons

    Parameters
    ----------
    geometries : array_like
        An array of polygons or coordinates (see polygons).
    indices : array_like or None
        Indices into the target array where input geometries belong. If
        provided, both geometries and indices should be 1D and have matching
        sizes.
    """
    typ = GeometryType.MULTIPOLYGON
    geometries = np.asarray(geometries)
    if not isinstance(geometries, Geometry) and np.issubdtype(
        geometries.dtype, np.number
    ):
        geometries = polygons(geometries)
    if indices is None:
        return lib.create_collection(geometries, typ)
    else:
        return collections_1d(geometries, indices, typ)


def geometrycollections(geometries, indices=None):
    """Create geometrycollections from arrays of geometries

    Parameters
    ----------
    geometries : array_like
        An array of geometries
    indices : array_like or None
        Indices into the target array where input geometries belong. If
        provided, both geometries and indices should be 1D and have matching
        sizes.
    """
    typ = GeometryType.GEOMETRYCOLLECTION
    if indices is None:
        return lib.create_collection(geometries, typ)
    else:
        return collections_1d(geometries, indices, typ)


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
