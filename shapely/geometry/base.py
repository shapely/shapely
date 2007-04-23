from ctypes import string_at \
    #, create_string_buffer \
    #, c_char_p, c_double, c_float, c_int, c_uint, c_size_t, c_ubyte \
    #, c_void_p, byref

from shapely.geos import lgeos

class BaseGeometry(object):
    
    """Defines methods common to all geometries.
    """

    def __init__(self):
        self._geom = None
        self._ndim = None
        self._crs = None

    def __del__(self):
        if self._geom is not None:
            lgeos.GEOSGeom_destroy(self._geom)

    def __str__(self):
        return self.towkt()

    def geometryType(self):
        """Returns a string representing the geometry type, e.g. 'Polygon'."""
        return string_at(lgeos.GEOSGeomType(self._geom))

    def towkt(self):
        """Returns a WKT string representation of the geometry."""
        return string_at(lgeos.GEOSGeomToWKT(self._geom))

    # Properties
    geom_type = property(geometryType)
    wkt = property(towkt)



