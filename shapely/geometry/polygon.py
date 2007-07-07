"""
"""

from ctypes import byref, c_double, c_int, c_void_p, cast, POINTER, pointer

from shapely.geos import lgeos
from shapely.geometry.base import BaseGeometry
from shapely.geometry.linestring import LineString


class LinearRing(LineString):

    """A linear ring.
    """

    _ndim = 2

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
        >>> ring = LinearRing( ((0.,0.),(0.,1.),(1.,1.),(1.,0.),(0.,0.)) )
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
            self._geom = lgeos.GEOSGeom_createLinearRing(cs)
            self._ndim = ndim


class Polygon(BaseGeometry):

    """A line string, also known as a polyline.
    
    """

    _exterior = None
    _interior = []
    _ndim = 2

    def __init__(self, exterior=None, interior=None):
        """Initialize.

        Parameters
        ----------
        exterior : sequence or array
            This may be an object that satisfies the numpy array protocol,
            providing an M x 2 or M x 3 (with z) array, or it may be a sequence
            of x, y (,z) coordinate sequences.

        Example
        -------
        >>> coords = ((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.))
        >>> polygon = Polygon(coords)
        """
        BaseGeometry.__init__(self)

        if exterior is not None:
            self._exterior = LinearRing(exterior)
            self._geom = lgeos.GEOSGeom_createPolygon(
                            c_void_p(self._exterior._geom),
                            POINTER(c_void_p)(),
                            0
                            )
            # Polygon geometry takes ownership of the ring
            self._exterior._owned = True
        if interior is not None:
            # TODO: interior rings. could be a pain in the neck thanks
            # GEOSGeom_createPolygon()
            raise NotImplementedError \
                , "interior rings are not possible in this version"

    @property
    def exterior(self):
        if self._exterior is None:
            # A polygon created from the abstract factory will have a null
            # _exterior attribute.
            ring = lgeos.GEOSGetExteriorRing(self._geom)
            self._exterior = LinearRing()
            self._exterior._geom = ring
            self._exterior._owned = True
        return self._exterior

    @property
    def interior(self):
        return self._interior

    @property
    def tuple(self):
        """Return a GeoJSON coordinate array."""
        return (self.exterior.tuple,)

    @property
    def ctypes(self):
        if not self._ctypes_data:
            self._ctypes_data = self.exterior.ctypes
        return self._ctypes_data

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        return {
            'version': 3,
            'shape': (len(self.exterior), self._ndim),
            'typestr': '<f8',
            'data': self.ctypes,
            }

# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

