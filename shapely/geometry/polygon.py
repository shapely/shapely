"""Polygons and their linear ring components
"""

import sys

if sys.version_info[0] < 3:
    range = xrange

from ctypes import c_double, c_void_p, cast, POINTER
from ctypes import ArgumentError
import weakref

from shapely.algorithms.cga import signed_area
from shapely.coords import required
from shapely.geos import lgeos
from shapely.geometry.base import BaseGeometry, geos_geom_from_py
from shapely.geometry.linestring import LineString, LineStringAdapter
from shapely.geometry.proxy import PolygonProxy

__all__ = ['Polygon', 'asPolygon', 'LinearRing', 'asLinearRing']


class LinearRing(LineString):
    """
    A closed one-dimensional feature comprising one or more line segments

    A LinearRing that crosses itself or touches itself at a single point is
    invalid and operations on it may fail.
    """
    
    def __init__(self, coordinates=None):
        """
        Parameters
        ----------
        coordinates : sequence
            A sequence of (x, y [,z]) numeric coordinate pairs or triples

        Rings are implicitly closed. There is no need to specific a final
        coordinate pair identical to the first.

        Example
        -------
        Construct a square ring.

          >>> ring = LinearRing( ((0, 0), (0, 1), (1 ,1 ), (1 , 0)) )
          >>> ring.is_closed
          True
          >>> ring.length
          4.0
        """
        BaseGeometry.__init__(self)
        if coordinates is not None:
            self._set_coords(coordinates)

    @property
    def __geo_interface__(self):
        return {
            'type': 'LinearRing',
            'coordinates': tuple(self.coords)
            }

    # Coordinate access

    _get_coords = BaseGeometry._get_coords

    def _set_coords(self, coordinates):
        self.empty()
        self._geom, self._ndim = geos_linearring_from_py(coordinates)

    coords = property(_get_coords, _set_coords)

    @property
    def is_ccw(self):
        """True is the ring is oriented counter clock-wise"""
        return bool(self.impl['is_ccw'](self))

    @property
    def is_simple(self):
        """True if the geometry is simple, meaning that any self-intersections
        are only at boundary points, else False"""
        return LineString(self).is_simple


class LinearRingAdapter(LineStringAdapter):

    __p__ = None

    def __init__(self, context):
        self.context = context
        self.factory = geos_linearring_from_py

    @property
    def __geo_interface__(self):
        return {
            'type': 'LinearRing',
            'coordinates': tuple(self.coords)
            }

    coords = property(BaseGeometry._get_coords)


def asLinearRing(context):
    """Adapt an object to the LinearRing interface"""
    return LinearRingAdapter(context)


class InteriorRingSequence(object):

    _factory = None
    _geom = None
    __p__ = None
    _ndim = None
    _index = 0
    _length = 0
    __rings__ = None
    _gtag = None

    def __init__(self, parent):
        self.__p__ = parent
        self._geom = parent._geom
        self._ndim = parent._ndim

    def __iter__(self):
        self._index = 0
        self._length = self.__len__()
        return self

    def __next__(self):
        if self._index < self._length:
            ring = self._get_ring(self._index)
            self._index += 1
            return ring
        else:
            raise StopIteration 

    if sys.version_info[0] < 3:
        next = __next__

    def __len__(self):
        return lgeos.GEOSGetNumInteriorRings(self._geom)

    def __getitem__(self, key):
        m = self.__len__()
        if isinstance(key, int):
            if key + m < 0 or key >= m:
                raise IndexError("index out of range")
            if key < 0:
                i = m + key
            else:
                i = key
            return self._get_ring(i)
        elif isinstance(key, slice):
            res = []
            start, stop, stride = key.indices(m)
            for i in range(start, stop, stride):
                res.append(self._get_ring(i))
            return res
        else:
            raise TypeError("key must be an index or slice")

    @property
    def _longest(self):
        max = 0
        for g in iter(self):
            l = len(g.coords)
            if l > max:
                max = l

    def gtag(self):
        return hash(repr(self.__p__))

    def _get_ring(self, i):
        gtag = self.gtag()
        if gtag != self._gtag:
            self.__rings__ = {}
        if i not in self.__rings__:
            g = lgeos.GEOSGetInteriorRingN(self._geom, i)
            ring = LinearRing()
            ring.__geom__ = g
            ring.__p__ = self
            ring._other_owned = True
            ring._ndim = self._ndim
            self.__rings__[i] = weakref.ref(ring)
        return self.__rings__[i]()
        

class Polygon(BaseGeometry):
    """
    A two-dimensional figure bounded by a linear ring

    A polygon has a non-zero area. It may have one or more negative-space
    "holes" which are also bounded by linear rings. If any rings cross each
    other, the feature is invalid and operations on it may fail.

    Attributes
    ----------
    exterior : LinearRing
        The ring which bounds the positive space of the polygon.
    interiors : sequence
        A sequence of rings which bound all existing holes.
    """

    _exterior = None
    _interiors = []
    _ndim = 2

    def __init__(self, shell=None, holes=None):
        """
        Parameters
        ----------
        shell : sequence
            A sequence of (x, y [,z]) numeric coordinate pairs or triples
        holes : sequence
            A sequence of objects which satisfy the same requirements as the
            shell parameters above

        Example
        -------
        Create a square polygon with no holes

          >>> coords = ((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.))
          >>> polygon = Polygon(coords)
          >>> polygon.area
          1.0
        """
        BaseGeometry.__init__(self)

        if shell is not None:
            self._geom, self._ndim = geos_polygon_from_py(shell, holes)

    @property
    def exterior(self):
        if self.is_empty:
            return None
        elif self._exterior is None or self._exterior() is None:
            g = lgeos.GEOSGetExteriorRing(self._geom)
            ring = LinearRing()
            ring.__geom__ = g
            ring.__p__ = self
            ring._other_owned = True
            ring._ndim = self._ndim
            self._exterior = weakref.ref(ring)
        return self._exterior()

    @property
    def interiors(self):
        if self.is_empty:
            return []
        return InteriorRingSequence(self)

    @property
    def ctypes(self):
        if not self._ctypes_data:
            self._ctypes_data = self.exterior.ctypes
        return self._ctypes_data

    @property
    def __array_interface__(self):
        raise NotImplementedError(
        "A polygon does not itself provide the array interface. Its rings do.")

    def _get_coords(self):
        raise NotImplementedError(
        "Component rings have coordinate sequences, but the polygon does not")

    def _set_coords(self, ob):
        raise NotImplementedError(
        "Component rings have coordinate sequences, but the polygon does not")

    @property
    def coords(self):
        raise NotImplementedError(
        "Component rings have coordinate sequences, but the polygon does not")

    @property
    def __geo_interface__(self):
        coords = [tuple(self.exterior.coords)]
        for hole in self.interiors:
            coords.append(tuple(hole.coords))
        return {
            'type': 'Polygon',
            'coordinates': tuple(coords)
            }

    def svg(self, scale_factor=1.):
        """
        SVG representation of the geometry. Scale factor is multiplied by
        the size of the SVG symbol so it can be scaled consistently for a
        consistent appearance based on the canvas size.
        """
        exterior_coords = [["{0},{1}".format(*c) for c in self.exterior.coords]]
        interior_coords = [
            ["{0},{1}".format(*c) for c in interior.coords]
            for interior in self.interiors ]
        path = " ".join([
            "M {0} L {1} z".format(coords[0], " L ".join(coords[1:]))
            for coords in exterior_coords + interior_coords ])
        return """
            <g fill-rule="evenodd" fill="{2}" stroke="#555555" 
            stroke-width="{0}" opacity="0.6">
            <path d="{1}" />
            </g>""".format(
                2.*scale_factor, path, "#66cc99" if self.is_valid else "#ff3333")


class PolygonAdapter(PolygonProxy, Polygon):
    
    def __init__(self, shell, holes=None):
        self.shell = shell
        self.holes = holes
        self.context = (shell, holes)
        self.factory = geos_polygon_from_py

    @property
    def _ndim(self):
        try:
            # From array protocol
            array = self.shell.__array_interface__
            n = array['shape'][1]
            assert n == 2 or n == 3
            return n
        except AttributeError:
            # Fall back on list
            return len(self.shell[0])


def asPolygon(shell, holes=None):
    """Adapt objects to the Polygon interface"""
    return PolygonAdapter(shell, holes)

def orient(polygon, sign=1.0):
    s = float(sign)
    rings = []
    ring = polygon.exterior
    if signed_area(ring)/s >= 0.0:
        rings.append(ring)
    else:
        rings.append(list(ring.coords)[::-1])
    for ring in polygon.interiors:
        if signed_area(ring)/s <= 0.0:
            rings.append(ring)
        else:
            rings.append(list(ring.coords)[::-1])
    return Polygon(rings[0], rings[1:])

def geos_linearring_from_py(ob, update_geom=None, update_ndim=0):
    # If a LinearRing is passed in, clone it and return
    # If a LineString is passed in, clone the coord seq and return a LinearRing
    if isinstance(ob, LineString):
        if type(ob) == LinearRing:
            return geos_geom_from_py(ob)
        else:
            if ob.is_closed and len(ob.coords) >= 4:
                return geos_geom_from_py(ob, lgeos.GEOSGeom_createLinearRing)

    # If numpy is present, we use numpy.require to ensure that we have a
    # C-continguous array that owns its data. View data will be copied.
    ob = required(ob)
    try:
        # From array protocol
        array = ob.__array_interface__
        assert len(array['shape']) == 2
        m = array['shape'][0]
        n = array['shape'][1]
        if m < 3:
            raise ValueError(
                "A LinearRing must have at least 3 coordinate tuples")
        assert n == 2 or n == 3

        # Make pointer to the coordinate array
        if isinstance(array['data'], tuple):
            # numpy tuple (addr, read-only)
            cp = cast(array['data'][0], POINTER(c_double))
        else:
            cp = array['data']

        # Add closing coordinates to sequence?
        if cp[0] != cp[m*n-n] or cp[1] != cp[m*n-n+1]:
            M = m + 1
        else:
            M = m

        # Create a coordinate sequence
        if update_geom is not None:
            cs = lgeos.GEOSGeom_getCoordSeq(update_geom)
            if n != update_ndim:
                raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim)
        else:
            cs = lgeos.GEOSCoordSeq_create(M, n)

        # add to coordinate sequence
        for i in range(m):
            # Because of a bug in the GEOS C API, 
            # always set X before Y
            lgeos.GEOSCoordSeq_setX(cs, i, cp[n*i])
            lgeos.GEOSCoordSeq_setY(cs, i, cp[n*i+1])
            if n == 3:
                lgeos.GEOSCoordSeq_setZ(cs, i, cp[n*i+2])

        # Add closing coordinates to sequence?
        if M > m:        
            # Because of a bug in the GEOS C API, 
            # always set X before Y
            lgeos.GEOSCoordSeq_setX(cs, M-1, cp[0])
            lgeos.GEOSCoordSeq_setY(cs, M-1, cp[1])
            if n == 3:
                lgeos.GEOSCoordSeq_setZ(cs, M-1, cp[2])
            
    except AttributeError:
        # Fall back on list
        try:
            m = len(ob)
        except TypeError:  # Iterators, e.g. Python 3 zip
            ob = list(ob)
            m = len(ob)

        n = len(ob[0])
        if m < 3:
            raise ValueError(
                "A LinearRing must have at least 3 coordinate tuples")
        assert (n == 2 or n == 3)

        # Add closing coordinates if not provided
        if m == 3 or ob[0][0] != ob[-1][0] or ob[0][1] != ob[-1][1]:
            M = m + 1
        else:
            M = m

        # Create a coordinate sequence
        if update_geom is not None:
            cs = lgeos.GEOSGeom_getCoordSeq(update_geom)
            if n != update_ndim:
                raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim)
        else:
            cs = lgeos.GEOSCoordSeq_create(M, n)
        
        # add to coordinate sequence
        for i in range(m):
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

        # Add closing coordinates to sequence?
        if M > m:
            coords = ob[0]
            # Because of a bug in the GEOS C API, 
            # always set X before Y
            lgeos.GEOSCoordSeq_setX(cs, M-1, coords[0])
            lgeos.GEOSCoordSeq_setY(cs, M-1, coords[1])
            if n == 3:
                lgeos.GEOSCoordSeq_setZ(cs, M-1, coords[2])

    if update_geom is not None:
        return None
    else:
        return lgeos.GEOSGeom_createLinearRing(cs), n

def update_linearring_from_py(geom, ob):
    geos_linearring_from_py(ob, geom._geom, geom._ndim)

def geos_polygon_from_py(shell, holes=None):
    if isinstance(shell, Polygon):
        return geos_geom_from_py(shell)

    if shell is not None:
        geos_shell, ndim = geos_linearring_from_py(shell)
        if holes is not None and len(holes) > 0:
            ob = holes
            L = len(ob)
            exemplar = ob[0]
            try:
                N = len(exemplar[0])
            except TypeError:
                N = exemplar._ndim
            if not L >= 1:
                raise ValueError("number of holes must be non zero")
            if not N in (2, 3):
                raise ValueError("insufficiant coordinate dimension")

            # Array of pointers to ring geometries
            geos_holes = (c_void_p * L)()
    
            # add to coordinate sequence
            for l in range(L):
                geom, ndim = geos_linearring_from_py(ob[l])
                geos_holes[l] = cast(geom, c_void_p)
        else:
            geos_holes = POINTER(c_void_p)()
            L = 0
        return (
            lgeos.GEOSGeom_createPolygon(
                        c_void_p(geos_shell),
                        geos_holes,
                        L
                        ),
            ndim
            )

# Test runner
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
