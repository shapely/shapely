
from shapely.geos import lgeos
from shapely.geometry.base import BaseGeometry
from shapely.geometry.point import Point

from ctypes import string_at

CLASS_TYPE = {
    'Point': Point
    #, LineString, LinearRing, Polygon, MultiPoint,
              #MultiLineString, MultiPolygon, GeometryCollection]
}

def factory(geom):
    """x."""
    ob = BaseGeometry()
    ob.__class__ = CLASS_TYPE[string_at(lgeos.GEOSGeomType(geom))]
    ob._geom = geom
    return ob

def json_factory(ob):
    try:
        coords = ob.get('coordinates', [])
        return CLASS_TYPE[ob.get('type')](*coords)
    except (KeyError, TypeError):
        pass
    return ob

