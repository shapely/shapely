"""Line strings and related utilities
"""

from ctypes import c_double, cast, POINTER
from ctypes import ArgumentError

from shapely.geos import lgeos, TopologicalError
from shapely.geometry.base import BaseGeometry, geom_factory
from shapely.geometry.proxy import CachingGeometryProxy

__all__ = ['LineString', 'asLineString']


class LineString(BaseGeometry):
    """
    A one-dimensional figure comprising one or more line segments
    
    A LineString has non-zero length and zero area. It may approximate a curve
    and need not be straight. Unlike a LinearRing, a LineString is not closed.
    """

    def __init__(self, coordinates=None):
        """
        Parameters
        ----------
        coordinates : sequence
            A sequence of (x, y [,z]) numeric coordinate pairs or triples or
            an object that provides the numpy array interface, including another
            instance of LineString.

        Example
        -------
        Create a line with two segments

          >>> a = LineString([[0, 0], [1, 0], [1, 1]])
          >>> a.length
          2.0
        """
        BaseGeometry.__init__(self)
        if coordinates is not None:
            self._set_coords(coordinates)

    @property
    def __geo_interface__(self):
        return {
            'type': 'LineString',
            'coordinates': tuple(self.coords)
            }

    @property
    def ctypes(self):
        if not self._ctypes_data:
            self._ctypes_data = self.coords.ctypes
        return self._ctypes_data

    def array_interface(self):
        """Provide the Numpy array protocol."""
        return self.coords.array_interface()
    
    __array_interface__ = property(array_interface)

    # Coordinate access
    def _set_coords(self, coordinates):
        self.empty()
        self._geom, self._ndim = geos_linestring_from_py(coordinates)

    coords = property(BaseGeometry._get_coords, _set_coords)

    @property
    def xy(self):
        """Separate arrays of X and Y coordinate values
        
        Example:
        
          >>> x, y = LineString(((0, 0), (1, 1))).xy
          >>> list(x)
          [0.0, 1.0]
          >>> list(y)
          [0.0, 1.0]
        """
        return self.coords.xy
    
    def parallel_offset(
        self, distance, side, 
        resolution=16, join_style=1, mitre_limit=1.0):
        
        """Returns a LineString or MultiLineString geometry at a distance from
        the object on its right or its left side.
        
        Distance must be a positive float value. The side parameter may be
        'left' or 'right'. The resolution of the buffer around each vertex of
        the object increases by increasing the resolution keyword parameter or
        third positional parameter.
        
        The join style is for outside corners between line segments. Accepted
        values are 1 => ROUND, 2 => MITRE, 3 => BEVEL.
        
        The mitre ratio limit is used for very sharp corners. It is the ratio
        of the distance from the corner to the end of the mitred offset corner.
        When two line segments meet at a sharp angle, a miter join will extend
        far beyond the original geometry. To prevent unreasonable geometry, the
        mitre limit allows controlling the maximum length of the join corner.
        Corners with a ratio which exceed the limit will be beveled."""

        try:
            return geom_factory(self.impl['parallel_offset'](
                self, distance, resolution, join_style, mitre_limit, 
                bool(side=='left')))
        except WindowsError:
            raise TopologicalError()
        

class LineStringAdapter(CachingGeometryProxy, LineString):

    def __init__(self, context):
        self.context = context
        self.factory = geos_linestring_from_py

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

    _get_coords = BaseGeometry._get_coords

    def _set_coords(self, ob):
        raise NotImplementedError(
            "Adapters can not modify their coordinate sources")

    coords = property(_get_coords)


def asLineString(context):
    """Adapt an object the LineString interface"""
    return LineStringAdapter(context)


def geos_linestring_from_py(ob, update_geom=None, update_ndim=0):
    try:
        # From array protocol
        array = ob.__array_interface__
        assert len(array['shape']) == 2
        m = array['shape'][0]
        if m < 2:
            raise ValueError(
                "LineStrings must have at least 2 coordinate tuples")
        try:
            n = array['shape'][1]
        except IndexError:
            raise ValueError(
                "Input %s is the wrong shape for a LineString" % str(ob))
        assert n == 2 or n == 3

        # Make pointer to the coordinate array
        if isinstance(array['data'], tuple):
            # numpy tuple (addr, read-only)
            cp = cast(array['data'][0], POINTER(c_double))
        else:
            cp = array['data']

        # Create a coordinate sequence
        if update_geom is not None:
            cs = lgeos.GEOSGeom_getCoordSeq(update_geom)
            if n != update_ndim:
                raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim)
        else:
            cs = lgeos.GEOSCoordSeq_create(m, n)

        # add to coordinate sequence
        for i in xrange(m):
            dx = c_double(cp[n*i])
            dy = c_double(cp[n*i+1])
            dz = None
            if n == 3:
                try:
                    dz = c_double(cp[n*i+2])
                except IndexError:
                    raise ValueError("Inconsistent coordinate dimensionality")

            # Because of a bug in the GEOS C API, 
            # always set X before Y
            lgeos.GEOSCoordSeq_setX(cs, i, dx)
            lgeos.GEOSCoordSeq_setY(cs, i, dy)
            if n == 3:
                lgeos.GEOSCoordSeq_setZ(cs, i, dz)

    except AttributeError:
        # Fall back on list
        m = len(ob)
        if m < 2:
            raise ValueError(
                "LineStrings must have at least 2 coordinate tuples")
        try:
            n = len(ob[0])
        except TypeError:
            raise ValueError(
                "Input %s is the wrong shape for a LineString" % str(ob))
        assert n == 2 or n == 3

        # Create a coordinate sequence
        if update_geom is not None:
            cs = lgeos.GEOSGeom_getCoordSeq(update_geom)
            if n != update_ndim:
                raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim)
        else:
            cs = lgeos.GEOSCoordSeq_create(m, n)
        
        # add to coordinate sequence
        for i in xrange(m):
            coords = ob[i]
            # Because of a bug in the GEOS C API, 
            # always set X before Y
            lgeos.GEOSCoordSeq_setX(cs, i, coords[0])
            lgeos.GEOSCoordSeq_setY(cs, i, coords[1])
            if n == 3:
                try:
                    lgeos.GEOSCoordSeq_setZ(cs, i, coords[2])
                except IndexError:
                    raise ValueError("Inconsistent coordinate dimensionality")
    
    if update_geom is not None:
        return None
    else:
        return lgeos.GEOSGeom_createLineString(cs), n

def update_linestring_from_py(geom, ob):
    geos_linestring_from_py(ob, geom._geom, geom._ndim)

# Test runner
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
