"""Collections of linestrings and related utilities
"""

from ctypes import c_double, c_void_p, cast, POINTER

from shapely.geos import lgeos
from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry.linestring import LineString, geos_linestring_from_py
from shapely.geometry.proxy import CachingGeometryProxy

__all__ = ['MultiLineString', 'asMultiLineString']


class MultiLineString(BaseMultipartGeometry):
    """
    A collection of one or more line strings
    
    A MultiLineString has non-zero length and zero area.

    Attributes
    ----------
    geoms : sequence
        A sequence of LineStrings
    """

    def __init__(self, lines=None):
        """
        Parameters
        ----------
        lines : sequence
            A sequence of line-like coordinate sequences or objects that
            provide the numpy array interface, including instances of
            LineString.

        Example
        -------
        Construct a collection containing one line string.

          >>> lines = MultiLineString( [[[0.0, 0.0], [1.0, 2.0]]] )
        """
        super(MultiLineString, self).__init__()

        if lines is None:
            # allow creation of empty multilinestrings, to support unpickling
            pass
        else:
            self._geom, self._ndim = geos_multilinestring_from_py(lines)

    def shape_factory(self, *args):
        return LineString(*args)

    @property
    def __geo_interface__(self):
        return {
            'type': 'MultiLineString',
            'coordinates': tuple(tuple(c for c in g.coords) for g in self.geoms)
            }


class MultiLineStringAdapter(CachingGeometryProxy, MultiLineString):
    
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
    """Adapts a sequence of objects to the MultiLineString interface"""
    return MultiLineStringAdapter(context)


def geos_multilinestring_from_py(ob):
    # ob must be either a sequence or array of sequences or arrays
    try:
        # From array protocol
        array = ob.__array_interface__
        assert len(array['shape']) == 1
        L = array['shape'][0]
        assert L >= 1

        # Array of pointers to sub-geometries
        subs = (c_void_p * L)()

        for l in xrange(L):
            geom, ndims = geos_linestring_from_py(array['data'][l])
            subs[i] = cast(geom, c_void_p)
        N = lgeos.GEOSGeom_getDimensions(subs[0])
    except (NotImplementedError, AttributeError):
        obs = getattr(ob, 'geoms', ob)
        L = len(obs)
        exemplar = obs[0]
        try:
            N = len(exemplar[0])
        except TypeError:
            N = exemplar._ndim
        assert L >= 1
        assert N == 2 or N == 3

        # Array of pointers to point geometries
        subs = (c_void_p * L)()
        
        # add to coordinate sequence
        for l in xrange(L):
            geom, ndims = geos_linestring_from_py(obs[l])
            subs[l] = cast(geom, c_void_p)
            
    return (lgeos.GEOSGeom_createCollection(5, subs, L), N)

# Test runner
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
