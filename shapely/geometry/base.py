"""
"""

from ctypes import string_at, byref, c_int, c_size_t, c_char_p, c_double

from shapely.geos import lgeos
from shapely.predicates import BinaryPredicate, UnaryPredicate
from shapely.topology import BinaryTopologicalOp, UnaryTopologicalOp


# Abstract geometry factory for use with topological methods below

def geom_factory(g):
    ob = BaseGeometry()
    geom_type = string_at(lgeos.GEOSGeomType(g))
    # TODO: check cost of dynamic import by profiling
    mod = __import__(
        'shapely.geometry', 
        globals(), 
        locals(), 
        [geom_type],
        )
    ob.__class__ = getattr(mod, geom_type)
    ob._geom = g
    return ob


class BaseGeometry(object):
    
    """Defines methods common to all geometries.
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

    def __reduce__(self):
        return (self.__class__, (), self.to_wkb())

    def __setstate__(self, state):
        self._geom = lgeos.GEOSGeomFromWKB_buf(
                        c_char_p(state), 
                        c_size_t(len(state))
                        )

    # Array and ctypes interfaces

    @property
    def tuple(self):
        """Return a GeoJSON coordinate array.
        
        To be overridden by extension classes."""
        raise NotImplemented

    @property
    def ctypes(self):
        """Return a ctypes representation.
        
        To be overridden by extension classes."""
        raise NotImplemented

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        raise NotImplemented

    # Python feature protocol

    @property
    def type(self):
        return self.geometryType()

    @property
    def coordinates(self):
        return self.tuple

    def geometryType(self):
        """Returns a string representing the geometry type, e.g. 'Polygon'."""
        return string_at(lgeos.GEOSGeomType(self._geom))

    def to_wkb(self):
        """Returns a WKT string representation of the geometry."""
        size = c_int()
        bytes = lgeos.GEOSGeomToWKB_buf(self._geom, byref(size))
        return string_at(bytes, size.value)

    def to_wkt(self):
        """Returns a WKT string representation of the geometry."""
        return string_at(lgeos.GEOSGeomToWKT(self._geom))

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

#extern int GEOS_DLL GEOSLength(const GEOSGeom g1, double *length);
#extern int GEOS_DLL GEOSDistance(const GEOSGeom g1, const GEOSGeom g2,
#	double *dist);

    # Properties
    geom_type = property(geometryType)
    wkt = property(to_wkt)
    wkb = property(to_wkb)

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

    # Relate Pattern (TODO?)
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

