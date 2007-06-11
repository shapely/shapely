
from shapely.geos import lgeos, ReadingError
from shapely.geometry.base import BaseGeometry
from shapely.geometry.point import Point

from ctypes import string_at, c_char_p, c_void_p, c_size_t

import geojson


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

def wkt_geometry(wkt):
    """Create a geometry from a well-known text string.

    Example
    -------
    >>> g = wkt_geometry('POINT (0.0 0.0)')
    >>> g.geom_type
    'Point'
    """
    geom = lgeos.GEOSGeomFromWKT(c_char_p(wkt))
    if not geom:
        raise ReadingError, \
        "Could not create geometry because of errors while reading input."
    return factory(geom)

def wkb_geometry(wkb):
    """Create a geometry from a well-known binary byte string."""
    geom = lgeos.GEOSGeomFromWKB_buf(c_char_p(wkb), c_size_t(len(wkb)));
    if not geom:
        raise ReadingError, \
        "Could not create geometry because of errors while reading input."
    return factory(geom)

def json_geometry(json):
    """Create a geometry from a GeoJSON object.
   
    TODO: use geojson package.
    """
    def factory(o):
        try:
            coords = o.get('coordinates', [])
            return Point(*coords)
        except (KeyError, TypeError):
            pass
        return o
    return geojson.loads(json, object_hook=factory)

