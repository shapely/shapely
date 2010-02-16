"""Multi-part collections of polygons and related utilities
"""

from ctypes import c_void_p, cast

from shapely.geos import lgeos
from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry.polygon import Polygon, geos_polygon_from_py
from shapely.geometry.proxy import CachingGeometryProxy

__all__ = ['MultiPolygon', 'asMultiPolygon']


def geos_multipolygon_from_py(ob):
    """ob must provide Python geo interface coordinates."""
    L = len(ob)
    N = len(ob[0][0][0])
    assert L >= 1
    assert N == 2 or N == 3

    subs = (c_void_p * L)()
    for l in xrange(L):
        geom, ndims = geos_polygon_from_py(ob[l][0], ob[l][1:])
        subs[l] = cast(geom, c_void_p)
            
    return (lgeos.GEOSGeom_createCollection(6, subs, L), N)

def geos_multipolygon_from_polygons(ob):
    """ob must be either a sequence or array of sequences or arrays."""
    L = len(ob)
    N = len(ob[0][0][0])
    assert L >= 1
    assert N == 2 or N == 3

    subs = (c_void_p * L)()
    for l in xrange(L):
        geom, ndims = geos_polygon_from_py(ob[l][0], ob[l][1])
        subs[l] = cast(geom, c_void_p)
            
    return (lgeos.GEOSGeom_createCollection(6, subs, L), N)


class MultiPolygon(BaseMultipartGeometry):

    """A two-dimensional figure comprising one or more polygons
    
    A MultiPolygon has non-zero area.
    
    Attributes
    ----------
    geoms : sequence
        A sequence of polygons
    """

    def __init__(self, polygons=None, context_type='polygons'):
        """Initialize a multipolygon instance

        Parameters
        ----------
        
        polygons : sequence
            A sequence of (shell, holes) tuples where shell is the sequence
            representation of a linear ring (see linearring.py) and holes is
            a sequence of such linear rings

        Example
        -------
          >>> ob = MultiPolygon( [
          ...     (
          ...     ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)), 
          ...     [((0.1,0.1), (0.1,0.2), (0.2,0.2), (0.2,0.1))]
          ...     )
          ... ] )
          >>> len(ob.geoms)
          1
          >>> type(ob.geoms[0]) == Polygon
          True

        """
        super(MultiPolygon, self).__init__()

        if polygons is None:
            # allow creation of null collections, to support unpickling
            pass
        elif context_type == 'polygons':
            self._geom, self._ndim = geos_multipolygon_from_polygons(polygons)
        elif context_type == 'geojson':
            self._geom, self._ndim = geos_multipolygon_from_py(polygons)

    def shape_factory(self, *args):
        return Polygon(*args)

    @property
    def __geo_interface__(self):
        allcoords = []
        for geom in self.geoms:
            coords = []
            coords.append(tuple(geom.exterior.coords))
            for hole in geom.interiors:
                coords.append(tuple(hole.coords))
            allcoords.append(coords)
        return {
            'type': 'MultiPolygon',
            'coordinates': allcoords
            }


class MultiPolygonAdapter(CachingGeometryProxy, MultiPolygon):

    """Adapts sequences of sequences or numpy arrays to the multipolygon
    interface.
    """
    
    context = None
    _owned = False

    def __init__(self, context, context_type='polygons'):
        self.context = context
        if context_type == 'geojson':
            self.factory = geos_multipolygon_from_py
        elif context_type == 'polygons':
            self.factory = geos_multipolygon_from_polygons

    @property
    def _ndim(self):
        try:
            # From array protocol
            array = self.context[0][0].__array_interface__
            n = array['shape'][1]
            assert n == 2 or n == 3
            return n
        except AttributeError:
            # Fall back on list
            return len(self.context[0][0][0])


def asMultiPolygon(context):
    """Factory for MultiLineStringAdapter instances."""
    return MultiPolygonAdapter(context)


# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()
