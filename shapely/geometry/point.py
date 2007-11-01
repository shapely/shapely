"""
Points.
"""

from ctypes import string_at, create_string_buffer, \
    c_char_p, c_double, c_float, c_int, c_uint, c_size_t, c_ubyte, \
    c_void_p, byref
from ctypes import cast, POINTER

from shapely.geos import lgeos, DimensionError
from shapely.geometry.base import BaseGeometry, CoordinateSequence


def geos_point_from_py(ob, update_geom=None, update_ndim=0):
    """Create a GEOS geom from an object that is a coordinate sequence
    or that provides the array interface.

    Returns the GEOS geometry and the number of its dimensions.
    """
    try:
        # From array protocol
        array = ob.__array_interface__
        assert len(array['shape']) == 1
        n = array['shape'][0]
        assert n == 2 or n == 3

        cdata = array['data'][0]
        cp = cast(cdata, POINTER(c_double))
        dx = c_double(cp[0])
        dy = c_double(cp[1])
        dz = None
        if n == 3:
            dz = c_double(cp[2])
            ndim = 3
    except AttributeError:
        # Fall back on list
        coords = ob
        n = len(coords)
        dx = c_double(coords[0])
        dy = c_double(coords[1])
        dz = None
        if n == 3:
            dz = c_double(coords[2])

    if update_geom:
        cs = lgeos.GEOSGeom_getCoordSeq(update_geom)
        if n != update_ndim:
            raise ValueError, \
            "Wrong coordinate dimensions; this geometry has dimensions: %d" \
            % update_ndim
    else:
        cs = lgeos.GEOSCoordSeq_create(1, n)
    
    # Because of a bug in the GEOS C API, always set X before Y
    lgeos.GEOSCoordSeq_setX(cs, 0, dx)
    lgeos.GEOSCoordSeq_setY(cs, 0, dy)
    if n == 3:
        lgeos.GEOSCoordSeq_setZ(cs, 0, dz)
   
    if update_geom:
        return None
    else:
        return (lgeos.GEOSGeom_createPoint(cs), n)

def update_point_from_py(geom, ob):
    geos_point_from_py(ob, geom._geom, geom._ndim)


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

    def __init__(self, *args):
        """This *copies* the given data to a new GEOS geometry.
        
        Parameters
        ----------
        
        There are 2 cases:

        1) 1 parameter: this must satisfy the numpy array protocol.
        2) 2 or more parameters: x, y, z : float
            Easting, northing, and elevation.
        """
        BaseGeometry.__init__(self)

        if len(args) == 0:
            # allow creation of null points, to support unpickling
            pass
        else:
            if len(args) == 1:
                self._geom, self._ndim = geos_point_from_py(args[0])
            else:
                self._geom, self._ndim = geos_point_from_py(tuple(args))

    # Coordinate getters and setters

    @property
    def x(self):
        """Return x coordinate."""
        cs = lgeos.GEOSGeom_getCoordSeq(self._geom)
        d = c_double()
        lgeos.GEOSCoordSeq_getX(cs, 0, byref(d))
        return d.value
    
    @property
    def y(self):
        """Return y coordinate."""
        cs = lgeos.GEOSGeom_getCoordSeq(self._geom)
        d = c_double()
        lgeos.GEOSCoordSeq_getY(cs, 0, byref(d))
        return d.value
    
    @property
    def z(self):
        """Return z coordinate."""
        if self._ndim != 3:
            raise DimensionError, "This point has no z coordinate."
        cs = lgeos.GEOSGeom_getCoordSeq(self._geom)
        d = c_double()
        lgeos.GEOSCoordSeq_getZ(cs, 0, byref(d))
        return d.value
    
    @property
    def __geo_interface__(self):
        return {
            'type': 'Point',
            'coordinates': self.coords[0]
            }

    @property
    def ctypes(self):
        if not self._ctypes_data:
            array_type = c_double * self._ndim
            array = array_type()
            array[0] = self.x
            array[1] = self.y
            if self._ndim == 3:
                array[2] = self.z
            self._ctypes_data = array
        return self._ctypes_data

    def array_interface(self):
        """Provide the Numpy array protocol."""
        ai = self.array_interface_base
        ai.update({'shape': (self._ndim,)})
        return ai
    __array_interface__ = property(array_interface)

    @property
    def bounds(self):
        cs = lgeos.GEOSGeom_getCoordSeq(self._geom)
        x = c_double()
        y = c_double()
        lgeos.GEOSCoordSeq_getX(cs, 0, byref(x))
        lgeos.GEOSCoordSeq_getY(cs, 0, byref(y))
        return (x.value, y.value, x.value, y.value)

    # Coordinate access

    def set_coords(self, coordinates):
        update_point_from_py(self, coordinates)

    coords = property(BaseGeometry.get_coords, set_coords)


class PointAdapter(Point):

    """Adapts a Python coordinate pair or a numpy array to the point interface.
    """

    context = None

    def __init__(self, context):
        self.context = context

    # Override base class __del__
    def __del__(self):
        pass

    @property
    def _ndim(self):
        try:
            # From array protocol
            array = self.context.__array_interface__
            n = array['shape'][0]
            assert n == 2 or n == 3
            return n
        except AttributeError:
            # Fall back on list
            return len(self.context)

    @property
    def _geom(self):
        """Keeps the GEOS geometry in synch with the context."""
        return geos_point_from_py(self.context)[0]       

    # TODO: reimplement x, y, z properties without calling invoking _geom

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        try:
            return self.context.__array_interface__
        except AttributeError:
            return self.array_interface()

    coords = property(BaseGeometry.get_coords)


def asPoint(context):
    """Factory for PointAdapter instances."""
    return PointAdapter(context)


# Test runner
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()

