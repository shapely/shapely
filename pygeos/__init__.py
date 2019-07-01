import numpy as np
from functools import wraps
from . import ufuncs
from .ufuncs import GEOSGeometry, GEOSException
from .ufuncs import *  # NoQA
from shapely import geometry as sg


def to_shapely(obj):
    shapely_geometry = sg.base.geom_factory(obj.ptr)
    shapely_geometry.__geom__ = sg.base.geos_geom_from_py(shapely_geometry)[0]
    return shapely_geometry


def to_wkt(obj):
    return to_shapely(obj).wkt


def to_wkb(obj):
    return to_shapely(obj).wkb


def to_shapely_recurse(obj):
    if isinstance(obj, GEOSGeometry):
        return to_shapely(obj)
    elif hasattr(obj, '__iter__'):
        return [to_shapely_recurse(x) for x in obj]
    else:
        return obj


def wrap_shapely_constructor(func):
    @wraps(func)
    def f(*args, **kwargs):
        geom = func(*to_shapely_recurse(args), **kwargs)
        return GEOSGeometry(sg.base.geos_geom_from_py(geom)[0])
    return f


box = wrap_shapely_constructor(sg.box)
MultiPoint = wrap_shapely_constructor(sg.MultiPoint)
MultiLineString = wrap_shapely_constructor(sg.MultiLineString)
MultiPolygon = wrap_shapely_constructor(sg.MultiPolygon)
GeometryCollection = wrap_shapely_constructor(sg.GeometryCollection)


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
    if not isinstance(shells, GEOSGeometry) and \
            np.issubdtype(shells.dtype, np.number):
        shells = linearrings(shells)

    if holes is None:
        return ufuncs.polygons_without_holes(shells)

    holes = np.asarray(holes)
    if not isinstance(holes, GEOSGeometry) and \
            np.issubdtype(holes.dtype, np.number):
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
    rings = np.moveaxis(rings, (0, 1), (-2, -1))
    return polygons(rings)
