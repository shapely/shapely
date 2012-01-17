"""Collections of points and related utilities
"""

from ctypes import byref, c_double, c_void_p, cast, POINTER
from ctypes import ArgumentError

from shapely.geos import lgeos
from shapely.geometry.base import BaseMultipartGeometry, exceptNull
from shapely.geometry.point import Point, geos_point_from_py
from shapely.geometry.proxy import CachingGeometryProxy

__all__ = ['MultiPoint', 'asMultiPoint']


class MultiPoint(BaseMultipartGeometry):

    """A collection of one or more points

    A MultiPoint has zero area and zero length.

    Attributes
    ----------
    geoms : sequence
        A sequence of Points
    """

    def __init__(self, points=None):
        """
        Parameters
        ----------
        points : sequence
            A sequence of (x, y [,z]) numeric coordinate pairs or triples or a
            sequence of objects that implement the numpy array interface,
            including instaces of Point.

        Example
        -------
        Construct a 2 point collection

          >>> ob = MultiPoint([[0.0, 0.0], [1.0, 2.0]])
          >>> len(ob.geoms)
          2
          >>> type(ob.geoms[0]) == Point
          True
        """
        super(MultiPoint, self).__init__()

        if points is None:
            # allow creation of empty multipoints, to support unpickling
            pass
        else:
            self._geom, self._ndim = geos_multipoint_from_py(points)

    def shape_factory(self, *args):
        return Point(*args)

    @property
    def __geo_interface__(self):
        return {
            'type': 'MultiPoint',
            'coordinates': tuple([g.coords[0] for g in self.geoms])
            }

    @property
    @exceptNull
    def ctypes(self):
        if not self._ctypes_data:
            temp = c_double()
            n = self._ndim
            m = len(self.geoms)
            array_type = c_double * (m * n)
            data = array_type()
            for i in xrange(m):
                g = self.geoms[i]._geom    
                cs = lgeos.GEOSGeom_getCoordSeq(g)
                lgeos.GEOSCoordSeq_getX(cs, 0, byref(temp))
                data[n*i] = temp.value
                lgeos.GEOSCoordSeq_getY(cs, 0, byref(temp))
                data[n*i+1] = temp.value
                if n == 3: # TODO: use hasz
                    lgeos.GEOSCoordSeq_getZ(cs, 0, byref(temp))
                    data[n*i+2] = temp.value
            self._ctypes_data = data
        return self._ctypes_data

    @exceptNull
    def array_interface(self):
        """Provide the Numpy array protocol."""
        ai = self.array_interface_base
        ai.update({'shape': (len(self.geoms), self._ndim)})
        return ai
    __array_interface__ = property(array_interface)


class MultiPointAdapter(CachingGeometryProxy, MultiPoint):

    context = None
    _owned = False

    def __init__(self, context):
        self.context = context
        self.factory = geos_multipoint_from_py

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
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        try:
            return self.context.__array_interface__
        except AttributeError:
            return self.array_interface()


def asMultiPoint(context):
    """Adapt a sequence of objects to the MultiPoint interface"""
    return MultiPointAdapter(context)


def geos_multipoint_from_py(ob):
    try:
        # From array protocol
        array = ob.__array_interface__
        assert len(array['shape']) == 2
        m = array['shape'][0]
        n = array['shape'][1]
        assert m >= 1
        assert n == 2 or n == 3

        # Make pointer to the coordinate array
        if isinstance(array['data'], tuple):
            # numpy tuple (addr, read-only)
            cp = cast(array['data'][0], POINTER(c_double))
        else:
            cp = array['data']

        # Array of pointers to sub-geometries
        subs = (c_void_p * m)()

        for i in xrange(m):
            geom, ndims = geos_point_from_py(cp[n*i:n*i+2])
            subs[i] = cast(geom, c_void_p)

    except AttributeError:
        # Fall back on list
        m = len(ob)
        try:
            n = len(ob[0])
        except TypeError:
            n = ob[0]._ndim
        assert n == 2 or n == 3

        # Array of pointers to point geometries
        subs = (c_void_p * m)()
        
        # add to coordinate sequence
        for i in xrange(m):
            coords = ob[i]
            geom, ndims = geos_point_from_py(coords)
            subs[i] = cast(geom, c_void_p)
            
    return lgeos.GEOSGeom_createCollection(4, subs, m), n

# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()
