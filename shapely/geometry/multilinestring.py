"""Multi-part collections of linestrings and related utilities
"""

from ctypes import c_double, c_void_p, cast, POINTER

from shapely.geos import lgeos
from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry.linestring import LineString, geos_linestring_from_py
from shapely.geometry.proxy import CachingGeometryProxy

__all__ = ['MultiLineString', 'asMultiLineString']


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


class MultiLineString(BaseMultipartGeometry):

    """A one-dimensional figure comprising one or more line strings
    
    A MultiLineString has non-zero length and zero area.

    Attributes
    ----------
    geoms : sequence
        A sequence of LineStrings
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
        super(MultiLineString, self).__init__()

        if coordinates is None:
            # allow creation of null lines, to support unpickling
            pass
        else:
            self._geom, self._ndim = geos_multilinestring_from_py(coordinates)

    def shape_factory(self, *args):
        return LineString(*args)

    @property
    def __geo_interface__(self):
        return {
            'type': 'MultiLineString',
            'coordinates': tuple(tuple(c for c in g.coords) for g in self.geoms)
            }


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
