"""
"""

from ctypes import string_at, create_string_buffer, \
    c_char_p, c_double, c_float, c_int, c_uint, c_size_t, c_ubyte, \
    c_void_p, byref

from shapely.geos import lgeos, DimensionError
from base import BaseGeometry

class Point(BaseGeometry):

    """A point geometry.
    
    Attributes
    ----------
    x, y, z : float
        Coordinate values

    Example
    -------
    >>> p = Point(1.0, -1.0)
    >>> str(p)
    'POINT (1.0000000000000000 -1.0000000000000000)'
    >>> p.y = 0.0
    >>> p.y
    0.0
    >>> p.x
    1.0
    >>> p.array
    [[1.0, 0.0]]
    """

    _ctypes_data = None

    def __init__(self, x=None, y=None, z=None, crs=None):
        """Initialize a point.
        
        Parameters
        ----------
        x, y, z : float
            Easting, northing, and elevation.
        crs : string
            PROJ.4 representation of a coordinate system.
        """
        BaseGeometry.__init__(self)

        # allow creation of null points, to support unpickling
        if x == y == z == None:
            pass
        else:
            # check coordinate input
            dx = c_double(x)
            dy = c_double(y)
            try:
                dz = c_double(z)
                ndim = 3
            except TypeError:
                dz = None
                ndim = 2
    
            self._geom = None
            self._ndim = ndim
            self._crs = crs

            cs = lgeos.GEOSCoordSeq_create(1, ndim)
            # Because of a bug in the GEOS C API, always set X before Y
            lgeos.GEOSCoordSeq_setX(cs, 0, dx)
            lgeos.GEOSCoordSeq_setY(cs, 0, dy)
            if ndim == 3:
                lgeos.GEOSCoordSeq_setZ(cs, 0, dz)
        
            self._geom = lgeos.GEOSGeom_createPoint(cs)

    # Coordinate getters and setters

    def getX(self):
        """Return x coordinate."""
        cs = lgeos.GEOSGeom_getCoordSeq(self._geom)
        d = c_double()
        lgeos.GEOSCoordSeq_getX(cs, 0, byref(d))
        return d.value
    
    def setX(self, x):
        """Set x coordinate."""
        cs = lgeos.GEOSGeom_getCoordSeq(self._geom)

        # First because of a GEOS C API bug, save Y
        dy = c_double()
        lgeos.GEOSCoordSeq_getY(cs, 0, byref(dy))
        
        if self._ndim == 3:
            dz = c_double()
            lgeos.GEOSCoordSeq_getZ(cs, 0, byref(dz))
        
        lgeos.GEOSCoordSeq_setX(cs, 0, c_double(x))

        # Now, restore Y. Yuck.
        lgeos.GEOSCoordSeq_setY(cs, 0, dy)

        if self._ndim == 3:
            lgeos.GEOSCoordSeq_setZ(cs, 0, dz)
    
    def getY(self):
        """Return y coordinate."""
        cs = lgeos.GEOSGeom_getCoordSeq(self._geom)
        d = c_double()
        lgeos.GEOSCoordSeq_getY(cs, 0, byref(d))
        return d.value
    
    def setY(self, y):
        """Set y coordinate."""
        cs = lgeos.GEOSGeom_getCoordSeq(self._geom)

        if self._ndim == 3:
            # First because of a GEOS C API bug, save Z
            d = c_double()
            lgeos.GEOSCoordSeq_getZ(cs, 0, byref(d))
        
            lgeos.GEOSCoordSeq_setY(cs, 0, c_double(y))
        
            # Now, restore Z. Yuck.
            lgeos.GEOSCoordSeq_setZ(cs, 0, d)
        else:
            lgeos.GEOSCoordSeq_setY(cs, 0, c_double(y))

    def getZ(self):
        """Return z coordinate."""
        if self._ndim != 3:
            raise DimensionError, \
            "This point has no z coordinate."
        cs = lgeos.GEOSGeom_getCoordSeq(self._geom)
        d = c_double()
        lgeos.GEOSCoordSeq_getZ(cs, 0, byref(d))
        return d.value
    
    def setZ(self, z):
        """Set z coordinate."""
        if self._ndim != 3:
            raise DimensionError, \
            "This point has no z coordinate."
        cs = lgeos.GEOSGeom_getCoordSeq(self._geom)
        lgeos.GEOSCoordSeq_setZ(cs, 0, c_double(z))
    
    # Coordinate properties
    x = property(getX, setX)
    y = property(getY, setY)
    z = property(getZ, setZ)

    @property
    def array(self):
        """Return a GeoJSON coordinate array."""
        array = [self.x, self.y]
        if self._ndim == 3: # TODO: use hasz
            array.append(self.z)
        return array

    @property
    def ctypes(self):
        if not self._ctypes_data:
            if self._ndim == 3: # TODO: use hasz
                array = c_double * 3
                self._ctypes_data = array(self.x, self.y, self.z)
            else:
                array = c_double * 2
                self._ctypes_data = array(self.x, self.y, self.z)
        return self._ctypes_data

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        return {
            'version': 3,
            'shape': (self._ndim,),
            'typestr': '>f8',
            'data': self.ctypes
            }


# Test runner
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()

