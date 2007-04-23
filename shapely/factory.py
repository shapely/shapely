
from shapely.geos import lgeos, ReadingError
from shapely.geometry.base import BaseGeometry
from shapely.geometry.point import Point

from ctypes import string_at, c_char_p

CLASS_TYPE = {
    'Point': Point
    #, LineString, LinearRing, Polygon, MultiPoint,
              #MultiLineString, MultiPolygon, GeometryCollection]
}

def wkt_geometry(wkt, crs=None):
    """Create a geometry from a well-known text string.

    Parameters
    ----------
    wkt : string
        WKT string like 'POINT (-105.5 40.0)'
    srs : SpatialReference
        The spatial reference system of the geometry

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

    ob = BaseGeometry()
    ob.__class__ = CLASS_TYPE[string_at(lgeos.GEOSGeomType(geom))]
    ob._crs = crs
    ob._geom = geom
    return ob


