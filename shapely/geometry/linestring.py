"""
"""

from ctypes import c_double, cast, POINTER

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


# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

