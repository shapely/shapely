"""
Multi-part collection of linestrings.
"""

from ctypes import byref, c_double, c_int, c_void_p, cast, POINTER, pointer

from shapely.geos import lgeos
from shapely.geometry.base import BaseGeometry, GeometrySequence, exceptNull
from shapely.geometry.linestring import LineString, geos_linestring_from_py
from shapely.geometry.proxy import CachingGeometryProxy


def geos_multilinestring_from_py(ob):
    """ob must be either a sequence or array of sequences or arrays."""
    try:
        # From array protocol
        array = ob.__array_interface__
        assert len(array['shape']) == 1
        L = array['shape'][0]
        assert L >= 1

        # Make pointer to the coordinate array
        cp = cast(array['data'][0], POINTER(c_double))

        # Array of pointers to sub-geometries
        subs = (c_void_p * L)()

        for l in xrange(L):
            geom, ndims = geos_linestring_from_py(array['data'][l])
            subs[i] = cast(geom, c_void_p)
        N = lgeos.GEOSGeom_getDimensions(subs[0])

    except AttributeError:
        # Fall back on list
        L = len(ob)
        N = len(ob[0][0])
        assert L >= 1
        assert N == 2 or N == 3

        # Array of pointers to point geometries
        subs = (c_void_p * L)()
        
        # add to coordinate sequence
        for l in xrange(L):
            geom, ndims = geos_linestring_from_py(ob[l])
            subs[l] = cast(geom, c_void_p)
            
    return (lgeos.GEOSGeom_createCollection(5, subs, L), N)


class MultiLineString(BaseGeometry):

    """a multiple linestring geometry.
    """

    def __init__(self, coordinates=None):
        """Initialize.

        Parameters
        ----------
        
        coordinates : sequence
            Contains coordinate sequences or objects that provide the numpy
            array protocol, providing an M x 2 or M x 3 (with z) array.

        Example
        -------

        >>> geom = MultiLineString( [[[0.0, 0.0], [1.0, 2.0]]] )
        >>> geom = MultiLineString( [ array([[0.0, 0.0], [1.0, 2.0]]) ] )
        
        Each result in a collection containing one line string.
        """
        BaseGeometry.__init__(self)

        if coordinates is None:
            # allow creation of null lines, to support unpickling
            pass
        else:
            self._geom, self._ndim = geos_multilinestring_from_py(coordinates)

    @property
    def __geo_interface__(self):
        return {
            'type': 'MultiLineString',
            'coordinates': tuple(tuple(c for c in g.coords) for g in self.geoms)
            }

    @property
    def ctypes(self):
        raise NotImplementedError, \
        "Multi-part geometries have no ctypes representations"

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        raise NotImplementedError, \
        "Multi-part geometries do not themselves provide the array interface"

    def _get_coords(self):
        raise NotImplementedError, \
        "Component rings have coordinate sequences, but the polygon does not"

    def _set_coords(self, ob):
        raise NotImplementedError, \
        "Component rings have coordinate sequences, but the polygon does not"

    @property
    def coords(self):
        raise NotImplementedError, \
        "Multi-part geometries do not provide a coordinate sequence"

    @property
    @exceptNull
    def geoms(self):
        return GeometrySequence(self, LineString)


class MultiLineStringAdapter(CachingGeometryProxy, MultiLineString):

    """Adapts sequences of sequences or numpy arrays to the multilinestring
    interface.
    """
    
    context = None
    _owned = False

    def __init__(self, context):
        self.context = context
        self.factory = geos_multilinestring_from_py

    @property
    def _ndim(self):
        try:
            # From array protocol
            array = self.context[0].__array_interface__
            n = array['shape'][1]
            assert n == 2 or n == 3
            return n
        except AttributeError:
            # Fall back on list
            return len(self.context[0][0])


def asMultiLineString(context):
    """Factory for MultiLineStringAdapter instances."""
    return MultiLineStringAdapter(context)


# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()
