from ctypes import string_at, byref, c_int, c_size_t, c_char_p
    #, create_string_buffer \
    #, c_char_p, c_double, c_float, c_int, c_uint, c_size_t, c_ubyte \
    #, c_void_p, byref

from shapely.geos import lgeos, OperationError

class BaseGeometry(object):
    
    """Defines methods common to all geometries.
    """

    def __init__(self):
        self._geom = lgeos.GEOSGeomFromWKT(c_char_p('GEOMETRYCOLLECTION EMPTY'))
        self._ndim = None
        self._crs = None

    def __del__(self):
        if self._geom is not None:
            lgeos.GEOSGeom_destroy(self._geom)

    def __str__(self):
        return self.to_wkt()

    def __reduce__(self):
        return (self.__class__, (), self.to_wkb())

    def __setstate__(self, state):
        self._geom = lgeos.GEOSGeomFromWKB_buf(c_char_p(state), 
                                               c_size_t(len(state))
                                               );

    def geometryType(self):
        """Returns a string representing the geometry type, e.g. 'Polygon'."""
        return string_at(lgeos.GEOSGeomType(self._geom))

    def to_wkb(self):
        """Returns a WKT string representation of the geometry."""
        size = c_int()
        bytes = lgeos.GEOSGeomToWKB_buf(self._geom, byref(size))
        return string_at(bytes, size.value)

    def to_wkt(self):
        """Returns a WKT string representation of the geometry."""
        return string_at(lgeos.GEOSGeomToWKT(self._geom))

    def equals(self, other):
        """Return True if self and other are equivalent, coordinate-wise."""
        result = lgeos.GEOSEquals(self._geom, other._geom)
        if result == 2:
            raise OperationError, "Failed to evaluate equals()"
        return bool(result)

    # Properties
    geom_type = property(geometryType)
    wkt = property(to_wkt)
    wkb = property(to_wkb)



