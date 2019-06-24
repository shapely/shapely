from functools import wraps
from .ufuncs import *  # NoQA
from shapely.geometry import \
    Point, LineString, LinearRing, Polygon, MultiPoint, MultiLineString,\
    MultiPolygon, GeometryCollection, box


def wrap_shapely_constructor(func):
    @wraps(func)
    def f(*args, **kwargs):
        geom = func(*args, **kwargs)
        return BaseGeometry(geom.__geom__)
    return f

box = wrap_shapely_constructor(box)
Point = wrap_shapely_constructor(Point)
LineString = wrap_shapely_constructor(LineString)
LinearRing = wrap_shapely_constructor(LinearRing)
Polygon = wrap_shapely_constructor(Polygon)
MultiPoint = wrap_shapely_constructor(MultiPoint)
MultiLineString = wrap_shapely_constructor(MultiLineString)
MultiPolygon = wrap_shapely_constructor(MultiPolygon)
GeometryCollection = wrap_shapely_constructor(GeometryCollection)
