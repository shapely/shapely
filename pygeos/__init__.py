from functools import wraps
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
Point = wrap_shapely_constructor(sg.Point)
LineString = wrap_shapely_constructor(sg.LineString)
LinearRing = wrap_shapely_constructor(sg.LinearRing)
Polygon = wrap_shapely_constructor(sg.Polygon)
MultiPoint = wrap_shapely_constructor(sg.MultiPoint)
MultiLineString = wrap_shapely_constructor(sg.MultiLineString)
MultiPolygon = wrap_shapely_constructor(sg.MultiPolygon)
GeometryCollection = wrap_shapely_constructor(sg.GeometryCollection)
