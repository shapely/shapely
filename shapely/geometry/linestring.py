"""
"""
import sys
from ctypes import byref, c_double, c_int, cast, POINTER, pointer

from shapely.geos import lgeos
from shapely.geometry.base import BaseGeometry

class LineString(BaseGeometry):

    """A line string, also known as a polyline.
    
    """

    def __init__(self, coordinates=None):
        """Initialize.

        Parameters
        ----------
        
        coordinates : sequence or array
            This may be an object that satisfies the numpy array protocol,
            providing an M x 2 or M x 3 (with z) array, or it may be a sequence
            of x, y (,z) coordinate sequences.

        Example
        -------

        >>> line = LineString([[0.0, 0.0], [1.0, 2.0]])
        >>> line = LineString(array([[0.0, 0.0], [1.0, 2.0]]))
        
        Each result in a line string from (0.0, 0.0) to (1.0, 2.0).
        """
        BaseGeometry.__init__(self)

        if coordinates is None:
            # allow creation of null lines, to support unpickling
            pass
        else:
            try:
                # From array protocol
                array = coordinates.__array_interface__
                
                # Check for proper shape
                m = array['shape'][0]
                n = array['shape'][1]
                assert m >= 2
                assert n == 2 or n == 3

                # Make pointer to the coordinate array
                cp = cast(array['data'][0], POINTER(c_double))

                # Create a coordinate sequence
                cs = lgeos.GEOSCoordSeq_create(m, n)

                # add to coordinate sequence
                for i in xrange(m):
                    dx = c_double(cp[n*i])
                    dy = c_double(cp[n*i+1])
                    dz = None
                    if n == 3:
                        dz = c_double(cp[n*i+2])
                
                    # Because of a bug in the GEOS C API, 
                    # always set X before Y
                    lgeos.GEOSCoordSeq_setX(cs, i, dx)
                    lgeos.GEOSCoordSeq_setY(cs, i, dy)
                    if n == 3:
                        lgeos.GEOSCoordSeq_setZ(cs, i, dz)
                ndim = n
            except AttributeError:
                # Fall back on list
                m = len(coordinates)
                n = len(coordinates[0])
                assert n == 2 or n == 3

                # Create a coordinate sequence
                cs = lgeos.GEOSCoordSeq_create(m, n)
                
                # add to coordinate sequence
                for i in xrange(m):
                    coords = coordinates[i]
                    dx = c_double(coords[0])
                    dy = c_double(coords[1])
                    dz = None
                    if n == 3:
                        dz = c_double(coords[2])
                
                    # Because of a bug in the GEOS C API, 
                    # always set X before Y
                    lgeos.GEOSCoordSeq_setX(cs, i, dx)
                    lgeos.GEOSCoordSeq_setY(cs, i, dy)
                    if n == 3:
                        lgeos.GEOSCoordSeq_setZ(cs, i, dz)
                ndim = n

            # Set geometry from coordinate string
            self._geom = lgeos.GEOSGeom_createLineString(cs)
            self._ndim = ndim

    def __len__(self):
        cs = lgeos.GEOSGeom_getCoordSeq(self._geom)
        cs_len = c_int(0)
        lgeos.GEOSCoordSeq_getSize(cs, byref(cs_len))
        return cs_len.value
        
    @property
    def array(self):
        """Return a GeoJSON coordinate array."""
        #cs = lgeos.GEOSGeom_getCoordSeq(self._geom)
        #cs_len = c_int(0)
        #lgeos.GEOSCoordSeq_getSize(cs, byref(cs_len))
        m = len(self)
        dx = c_double()
        dy = c_double()
        dz = c_double()
        array = []
        for i in xrange(cs_len.value):
            lgeos.GEOSCoordSeq_getX(cs, i, byref(dx))
            lgeos.GEOSCoordSeq_getY(cs, i, byref(dy))
            coords = [dx.value, dy.value]
            if self._ndim == 3: # TODO: use hasz
                lgeos.GEOSCoordSeq_getZ(cs, i, byref(dz))
                coords.append(dz.value)
            array.append(coords)
        return array

    @property
    def ctypes(self):
        if not self._ctypes_data:
            cs = lgeos.GEOSGeom_getCoordSeq(self._geom)
            cs_len = c_int(0)
            lgeos.GEOSCoordSeq_getSize(cs, byref(cs_len))
            temp = c_double()
            n = self._ndim
            m = cs_len.value
            array_type = c_double * (m * n)
            data = array_type()
            for i in xrange(m):
                lgeos.GEOSCoordSeq_getX(cs, i, byref(temp))
                data[n*i] = temp.value
                lgeos.GEOSCoordSeq_getY(cs, i, byref(temp))
                data[n*i+1] = temp.value
                if n == 3: # TODO: use hasz
                    lgeos.GEOSCoordSeq_getZ(cs, i, byref(temp))
                    data[n*i+2] = temp.value
            self._ctypes_data = data
        return self._ctypes_data

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        return {
            'version': 3,
            'shape': (len(self), self._ndim),
            'typestr': '<f8',
            'data': self.ctypes,
            }

# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

