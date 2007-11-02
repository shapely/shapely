"""
Base geometry class and utilities.
"""

from ctypes import string_at, byref, c_int, c_size_t, c_char_p, c_double
import sys

from shapely.geos import lgeos
from shapely.predicates import BinaryPredicate, UnaryPredicate
from shapely.topology import BinaryTopologicalOp, UnaryTopologicalOp


GEOMETRY_TYPES = [
    'Point',
	'LineString',
	'LinearRing',
	'Polygon',
	'MultiPoint',
	'MultiLineString',
	'MultiPolygon',
	'GeometryCollection'
    ]

def geometry_type_name(g):
    return GEOMETRY_TYPES[lgeos.GEOSGeomTypeId(g)]

# Abstract geometry factory for use with topological methods below

def geom_factory(g):
    if not g:
        raise ValueError, "No Shapely geometry can be created from this null value"
    ob = BaseGeometry()
    geom_type = geometry_type_name(g)
    # TODO: check cost of dynamic import by profiling
    mod = __import__(
        'shapely.geometry', 
        globals(), 
        locals(), 
        [geom_type],
        )
    ob.__class__ = getattr(mod, geom_type)
    ob._geom = g
    ob._ndim = 2 # callers should be all from 2D worlds
    return ob


class CoordinateSequence(object):

    _cseq = None
    _ndim = None
    _length = 0
    index = 0

    def __init__(self, geom):
        self._cseq = lgeos.GEOSGeom_getCoordSeq(geom._geom)
        self._ndim = geom._ndim

    def __iter__(self):
        self.index = 0
        self._length = self.__len__()
        return self

    def next(self):
        dx = c_double()
        dy = c_double()
        dz = c_double()
        i = self.index
        if i < self._length:
            lgeos.GEOSCoordSeq_getX(self._cseq, i, byref(dx))
            lgeos.GEOSCoordSeq_getY(self._cseq, i, byref(dy))
            if self._ndim == 3: # TODO: use hasz
                lgeos.GEOSCoordSeq_getZ(self._cseq, i, byref(dz))
                self.index += 1
                return (dx.value, dy.value, dz.value)
            else:
                self.index += 1
                return (dx.value, dy.value)
        else:
            raise StopIteration 

    def __len__(self):
        cs_len = c_int(0)
        lgeos.GEOSCoordSeq_getSize(self._cseq, byref(cs_len))
        return cs_len.value
    
    def __getitem__(self, i):
        M = self.__len__()
        if i + M < 0 or i >= M:
            raise IndexError, "index out of range"
        if i < 0:
            ii = M + i
        else:
            ii = i
        dx = c_double()
        dy = c_double()
        dz = c_double()
        lgeos.GEOSCoordSeq_getX(self._cseq, ii, byref(dx))
        lgeos.GEOSCoordSeq_getY(self._cseq, ii, byref(dy))
        if self._ndim == 3: # TODO: use hasz
            lgeos.GEOSCoordSeq_getZ(self._cseq, ii, byref(dz))
            return (dx.value, dy.value, dz.value)
        else:
            return (dx.value, dy.value)


class GeometrySequence(object):

    _factory = None
    _geom = None
    _ndim = None
    _index = 0
    _length = 0

    def __init__(self, geom, type):
        self._factory = type
        self._geom = geom._geom
        self._ndim = geom._ndim

    def __iter__(self):
        self._index = 0
        self._length = self.__len__()
        return self

    def next(self):
        if self._index < self.__len__():
            g = self._factory()
            g._owned = True
            g._geom = lgeos.GEOSGetGeometryN(self._geom, self._index)
            self._index += 1
            return g
        else:
            raise StopIteration 

    def __len__(self):
        return lgeos.GEOSGetNumGeometries(self._geom)

    def __getitem__(self, i):
        M = self.__len__()
        if i + M < 0 or i >= M:
            raise IndexError, "index out of range"
        if i < 0:
            ii = M + i
        else:
            ii = i
        g = self._factory()
        g._owned = True
        g._geom = lgeos.GEOSGetGeometryN(self._geom, ii)
        return g

    @property
    def _longest(self):
        max = 0
        for g in iter(self):
            l = len(g.coords)
            if l > max:
                max = l


class BaseGeometry(object):
    
    """Provides GEOS spatial predicates and topological operations.
    """

    _geom = None
    _ctypes_data = None
    _ndim = None
    _crs = None
    _owned = False

    def __init__(self):
        self._geom = lgeos.GEOSGeomFromWKT(c_char_p('GEOMETRYCOLLECTION EMPTY'))

    def __del__(self):
        if self._geom is not None and not self._owned:
            lgeos.GEOSGeom_destroy(self._geom)

    def __str__(self):
        return self.to_wkt()

    def __eq__(self, other):
        return self.equals(other)
    
    def __ne__(self, other):
        return not self.equals(other)

    # To support pickling

    def __reduce__(self):
        return (self.__class__, (), self.to_wkb())

    def __setstate__(self, state):
        self._geom = lgeos.GEOSGeomFromWKB_buf(
                        c_char_p(state), 
                        c_size_t(len(state))
                        )

    # Array and ctypes interfaces

    @property
    def ctypes(self):
        """Return a ctypes representation.
        
        To be overridden by extension classes."""
        raise NotImplementedError

    @property
    def array_interface_base(self):
        if sys.byteorder == 'little':
            typestr = '<f8'
        elif sys.byteorder == 'big':
            typestr = '>f8'
        else:
            raise ValueError, "Unsupported byteorder: neither little nor big-endian"
        return {
            'version': 3,
            'typestr': typestr,
            'data': self.ctypes,
            }

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        raise NotImplementedError

    def get_coords(self):
        return CoordinateSequence(self)

    def set_coords(self, ob):
        raise NotImplementedError, \
            "set_coords must be provided by derived classes"

    coords = property(get_coords, set_coords)

    # Python feature protocol

    @property
    def __geo_interface__(self):
        raise NotImplementedError

    @property
    def type(self):
        return self.geometryType()

    # Type of geometry and its representations

    def geometryType(self):
        """Returns a string representing the geometry type, e.g. 'Polygon'."""
        return geometry_type_name(self._geom)

    def to_wkb(self):
        """Returns a WKB byte string representation of the geometry."""
        size = c_int()
        bytes = lgeos.GEOSGeomToWKB_buf(self._geom, byref(size))
        return string_at(bytes, size.value)

    def to_wkt(self):
        """Returns a WKT string representation of the geometry."""
        return string_at(lgeos.GEOSGeomToWKT(self._geom))

    geom_type = property(geometryType)
    wkt = property(to_wkt)
    wkb = property(to_wkb)

    # Basic geometry properties

    @property
    def area(self):
        a = c_double()
        retval =  lgeos.GEOSArea(self._geom, byref(a))
        return a.value

    @property
    def length(self):
        len = c_double()
        retval =  lgeos.GEOSLength(self._geom, byref(len))
        return len.value

    def distance(self, other):
        d = c_double()
        retval =  lgeos.GEOSDistance(self._geom, other._geom, byref(d))
        return d.value

    # Topology operations
    #
    # These use descriptors to reduce the amount of boilerplate.
   
    envelope = UnaryTopologicalOp(lgeos.GEOSEnvelope, geom_factory)
    intersection = BinaryTopologicalOp(lgeos.GEOSIntersection, geom_factory)
    convex_hull = UnaryTopologicalOp(lgeos.GEOSConvexHull, geom_factory)
    difference = BinaryTopologicalOp(lgeos.GEOSDifference, geom_factory)
    symmetric_difference = BinaryTopologicalOp(lgeos.GEOSSymDifference, 
                                               geom_factory)
    boundary = UnaryTopologicalOp(lgeos.GEOSBoundary, geom_factory)
    union = BinaryTopologicalOp(lgeos.GEOSUnion, geom_factory)
    centroid = UnaryTopologicalOp(lgeos.GEOSGetCentroid, geom_factory)

    # Buffer has a unique distance argument, so not a descriptor
    def buffer(self, distance, quadsegs=16):
        return geom_factory(
            lgeos.GEOSBuffer(self._geom, c_double(distance), c_int(quadsegs))
            )

    # Relate has a unique string return value
    def relate(self, other):
        func = lgeos.GEOSRelate
        func.restype = c_char_p
        return lgeos.GEOSRelate(self._geom, other._geom)

    # Binary predicates
    #
    # These use descriptors to reduce the amount of boilerplate.

    # TODO: Relate Pattern?
    disjoint = BinaryPredicate(lgeos.GEOSDisjoint)
    touches = BinaryPredicate(lgeos.GEOSTouches)
    intersects = BinaryPredicate(lgeos.GEOSIntersects)
    crosses = BinaryPredicate(lgeos.GEOSCrosses)
    within = BinaryPredicate(lgeos.GEOSWithin)
    contains = BinaryPredicate(lgeos.GEOSContains)
    overlaps = BinaryPredicate(lgeos.GEOSOverlaps)
    equals = BinaryPredicate(lgeos.GEOSEquals)

    # Unary predicates
    #
    # These use descriptors to reduce the amount of boilerplate.

    is_empty = UnaryPredicate(lgeos.GEOSisEmpty)
    is_valid = UnaryPredicate(lgeos.GEOSisValid)
    is_simple = UnaryPredicate(lgeos.GEOSisSimple)
    is_ring = UnaryPredicate(lgeos.GEOSisRing)
    has_z = UnaryPredicate(lgeos.GEOSHasZ)

    @property
    def bounds(self):
        env = self.envelope
        if env.geom_type != 'Polygon':
            raise ValueError, env.wkt
        cs = lgeos.GEOSGeom_getCoordSeq(env.exterior._geom)
        cs_len = c_int(0)
        lgeos.GEOSCoordSeq_getSize(cs, byref(cs_len))
        
        minx = 1.e+20
        maxx = -1e+20
        miny = 1.e+20
        maxy = -1e+20
        temp = c_double()
        for i in xrange(cs_len.value):
            lgeos.GEOSCoordSeq_getX(cs, i, byref(temp))
            x = temp.value
            if x < minx: minx = x
            if x > maxx: maxx = x
            lgeos.GEOSCoordSeq_getY(cs, i, byref(temp))
            y = temp.value
            if y < miny: miny = y
            if y > maxy: maxy = y
        
        return (minx, miny, maxx, maxy)

