"""
Line strings.
"""

from ctypes import byref, c_double, c_int, cast, POINTER, pointer
from ctypes import ArgumentError

from shapely.geos import lgeos
from shapely.geometry.base import BaseGeometry


def geos_linestring_from_py(ob, update_geom=None, update_ndim=0):
    try:
        # From array protocol
        array = ob.__array_interface__
        assert len(array['shape']) == 2
        m = array['shape'][0]
        n = array['shape'][1]
        assert m >= 2
        assert n == 2 or n == 3

        # Make pointer to the coordinate array
        try:
            cp = cast(array['data'][0], POINTER(c_double))
        except ArgumentError:
            cp = array['data']

        # Create a coordinate sequence
        if update_geom is not None:
            cs = lgeos.GEOSGeom_getCoordSeq(update_geom)
            if n != update_ndim:
                raise ValueError, \
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim
        else:
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

    except AttributeError:
        # Fall back on list
        m = len(ob)
        n = len(ob[0])
        assert m >= 2
        assert n == 2 or n == 3

        # Create a coordinate sequence
        if update_geom is not None:
            cs = lgeos.GEOSGeom_getCoordSeq(update_geom)
            if n != update_ndim:
                raise ValueError, \
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim
        else:
            cs = lgeos.GEOSCoordSeq_create(m, n)
        
        # add to coordinate sequence
        for i in xrange(m):
            coords = ob[i]
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
    
    if update_geom is not None:
        return None
    else:
        return (lgeos.GEOSGeom_createLineString(cs), n)

def update_linestring_from_py(geom, ob):
    geos_linestring_from_py(ob, geom._geom, geom._ndim)


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
            self._geom, self._ndim = geos_linestring_from_py(coordinates)

    @property
    def __geo_interface__(self):
        return {
            'type': 'LineString',
            'coordinates': tuple(self.coords)
            }

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

    def array_interface(self):
        """Provide the Numpy array protocol."""
        ai = self.array_interface_base
        ai.update({'shape': (len(self.coords), self._ndim)})
        return ai
    __array_interface__ = property(array_interface)

    # Coordinate access

    def set_coords(self, coordinates):
        update_linestring_from_py(self, coordinates)

    coords = property(BaseGeometry.get_coords, set_coords)


class LineStringAdapter(LineString):

    """Adapts a Python coordinate pair or a numpy array to the line string
    interface.
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
            n = array['shape'][1]
            assert n == 2 or n == 3
            return n
        except AttributeError:
            # Fall back on list
            return len(self.context[0])

    @property
    def _geom(self):
        """Keeps the GEOS geometry in synch with the context."""
        return geos_linestring_from_py(self.context)[0]       

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        try:
            return self.context.__array_interface__
        except AttributeError:
            return self.array_interface()

    coords = property(BaseGeometry.get_coords)


def asLineString(context):
    """Factory for PointAdapter instances."""
    return LineStringAdapter(context)

    
# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

