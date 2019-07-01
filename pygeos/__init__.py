import numpy as np
from functools import wraps
from . import ufuncs
from .ufuncs import GEOSGeometry
from .ufuncs import *  # NoQA
from shapely import geometry as sg


def to_shapely(obj):
    shapely_geometry = sg.base.geom_factory(obj.ptr)
    shapely_geometry.__geom__ = sg.base.geos_geom_from_py(shapely_geometry)[0]
    return shapely_geometry


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


def points(coords, y=None, z=None):
    """Create an array of points.

    Attributes
    ----------
    coords : array_like
        An array of coordinate tuples (2- or 3-dimensional) or, if `y` is
        provided the x coordinates
    y : array_like
    z : array_like
    """
    if y is None:
        return ufuncs.points(coords)
    x = coords
    if z is None:
        coords = np.broadcast_arrays(x, y)
    else:
        coords = np.broadcast_arrays(x, y, z)
    return ufuncs.points(np.stack(coords, axis=-1))
