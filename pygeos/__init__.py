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
LineString = wrap_shapely_constructor(sg.LineString)
LinearRing = wrap_shapely_constructor(sg.LinearRing)
Polygon = wrap_shapely_constructor(sg.Polygon)
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

    The rings are not closed automatically.

    Attributes
    ----------
    coords : array_like
        An array of lists of coordinate tuples (2- or 3-dimensional) or, if `y`
        is provided, an array of lists of x coordinates
    y : array_like
    z : array_like
    """
    return _wrap_construct_ufunc(ufuncs.linearrings, coords, y, z)
