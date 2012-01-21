"""Base geometry class and utilities
"""

import sys
import warnings

from shapely.coords import CoordinateSequence
from shapely.ftools import wraps
from shapely.geos import lgeos
from shapely.impl import DefaultImplementation, delegated
from shapely import wkb, wkt

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
    if g is None:
        raise ValueError("Null geometry has no type")
    return GEOMETRY_TYPES[lgeos.GEOSGeomTypeId(g)]

def geom_factory(g, parent=None):
    # Abstract geometry factory for use with topological methods below
    if not g:
        raise ValueError("No Shapely geometry can be created from null value")
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
    ob.__geom__ = g
    ob.__p__ = parent
    if lgeos.methods['has_z'](g):
        ob._ndim = 3
    else:
        ob._ndim = 2
    return ob

def exceptNull(func):
    """Decorator which helps avoid GEOS operations on null pointers."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not args[0]._geom or args[0].is_empty:
            raise ValueError("Null/empty geometry supports no operations")
        return func(*args, **kwargs)
    return wrapper

EMPTY = wkb.deserialize('010700000000000000'.decode('hex'))

class BaseGeometry(object):
    """
    Provides GEOS spatial predicates and topological operations.

    """

    # Attributes
    # ----------
    # __geom__ : c_void_p
    #     Cached ctypes pointer to GEOS geometry. Not to be accessed.
    # _geom : c_void_p
    #     Property by which the GEOS geometry is accessed.
    # __p__ : object
    #     Parent (Shapely) geometry
    # _ctypes_data : object
    #     Cached ctypes data buffer
    # _ndim : int
    #     Number of dimensions (2 or 3, generally)
    # _crs : object
    #     Coordinate reference system. Available for Shapely extensions, but
    #     not implemented here.
    # _owned : bool
    #     True if this object's GEOS geometry is owned by another as in the case
    #     of a multipart geometry member.
    __geom__ = EMPTY
    __p__ = None
    _ctypes_data = None
    _ndim = None
    _crs = None
    _owned = False
    
    # Backend config
    impl = DefaultImplementation

    @property
    def _is_empty(self):
        return self.__geom__ in [EMPTY, None]

    # a reference to the so/dll proxy to preserve access during clean up
    _lgeos = lgeos

    def empty(self):
        # TODO: defer cleanup to the implementation. We shouldn't be
        # explicitly calling a lgeos method here.
        if not (self._owned or self._is_empty):
            try:
                self._lgeos.GEOSGeom_destroy(self.__geom__)
            except AttributeError:
                pass # _lgeos might be empty on shutdown
        self.__geom__ = EMPTY

    def __del__(self):
        self.empty()
        self.__geom__ = None
        self.__p__ = None

    def __str__(self):
        return self.to_wkt()

    # To support pickling
    def __reduce__(self):
        return (self.__class__, (), self.to_wkb())

    def __setstate__(self, state):
        self.empty()
        self.__geom__ = wkb.deserialize(state)
    
    # The _geom property
    def _get_geom(self):
        return self.__geom__
    def _set_geom(self, val):
        self.empty()
        self.__geom__ = val
    _geom = property(_get_geom, _set_geom)

    # Array and ctypes interfaces
    # ---------------------------

    @property
    def ctypes(self):
        """Return ctypes buffer"""
        raise NotImplementedError

    @property
    def array_interface_base(self):
        if sys.byteorder == 'little':
            typestr = '<f8'
        elif sys.byteorder == 'big':
            typestr = '>f8'
        else:
            raise ValueError(
                  "Unsupported byteorder: neither little nor big-endian")
        return {
            'version': 3,
            'typestr': typestr,
            'data': self.ctypes,
            }

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        raise NotImplementedError

    # Coordinate access
    # -----------------

    def _get_coords(self):
        """Access to geometry's coordinates (CoordinateSequence)"""
        if self.is_empty:
            return []
        return CoordinateSequence(self)

    def _set_coords(self, ob):
        raise NotImplementedError(
            "set_coords must be provided by derived classes")

    coords = property(_get_coords, _set_coords)

    @property
    def xy(self):
        """Separate arrays of X and Y coordinate values"""
        raise NotImplementedError

    # Python feature protocol

    @property
    def __geo_interface__(self):
        """Dictionary representation of the geometry"""
        raise NotImplementedError

    # Type of geometry and its representations
    # ----------------------------------------

    def geometryType(self):
        return geometry_type_name(self._geom)
    
    @property
    def type(self):
        return self.geometryType()

    def to_wkb(self):
        return wkb.dumps(self)

    def to_wkt(self):
        return wkt.dumps(self)

    geom_type = property(geometryType, 
        doc="""Name of the geometry's type, such as 'Point'"""
        )
    wkt = property(to_wkt,
        doc="""WKT representation of the geometry""")
    wkb = property(to_wkb,
        doc="""WKB representation of the geometry""")

    # Real-valued properties and methods
    # ----------------------------------

    @property
    def area(self):
        """Unitless area of the geometry (float)"""
        return self.impl['area'](self)

    def distance(self, other):
        """Unitless distance to other geometry (float)"""
        return self.impl['distance'](self, other)

    @property
    def length(self):
        """Unitless length of the geometry (float)"""
        return self.impl['length'](self)

    # Topological properties
    # ----------------------

    @property
    def boundary(self):
        """Returns a lower dimension geometry that bounds the object
        
        The boundary of a polygon is a line, the boundary of a line is a
        collection of points. The boundary of a point is an empty (null)
        collection.
        """
        return geom_factory(self.impl['boundary'](self))

    @property
    def bounds(self):
        """Returns minimum bounding region (minx, miny, maxx, maxy)"""
        if self.is_empty:
            return ()
        else:
            return self.impl['bounds'](self)
            
    @property
    def centroid(self):
        """Returns the geometric center of the object"""
        return geom_factory(self.impl['centroid'](self))

    @delegated
    def representative_point(self):
        """Returns a point guaranteed to be within the object, cheaply."""
        return geom_factory(self.impl['representative_point'](self))

    @property
    def convex_hull(self):
        """Imagine an elastic band stretched around the geometry: that's a 
        convex hull, more or less

        The convex hull of a three member multipoint, for example, is a
        triangular polygon.
        """ 
        return geom_factory(self.impl['convex_hull'](self))

    @property
    def envelope(self):
        """A figure that envelopes the geometry"""
        return geom_factory(self.impl['envelope'](self))

    def buffer(self, distance, resolution=16, quadsegs=None):
        """Returns a geometry with an envelope at a distance from the object's 
        envelope
        
        A negative distance has a "shrink" effect. A zero distance may be used
        to "tidy" a polygon. The resolution of the buffer around each vertex of
        the object increases by increasing the resolution keyword parameter
        or second positional parameter. Note: the use of a `quadsegs` parameter
        is deprecated and will be gone from the next major release.

        Example:

          >>> from shapely.wkt import loads
          >>> g = loads('POINT (0.0 0.0)')
          >>> g.buffer(1.0).area        # 16-gon approx of a unit radius circle
          3.1365484905459389
          >>> g.buffer(1.0, 128).area   # 128-gon approximation
          3.1415138011443009
          >>> g.buffer(1.0, 3).area     # triangle approximation
          3.0
        """
        if quadsegs is not None:
            warnings.warn(
                "The `quadsegs` argument is deprecated. Use `resolution`.", 
                DeprecationWarning)
            res = quadsegs
        else:
            res = resolution
        return geom_factory(self.impl['buffer'](self, distance, res))

    @delegated
    def simplify(self, tolerance, preserve_topology=True):
        """Returns a simplified geometry produced by the Douglas-Puecker 
        algorithm

        Coordinates of the simplified geometry will be no more than the
        tolerance distance from the original. Unless the topology preserving
        option is used, the algorithm may produce self-intersecting or
        otherwise invalid geometries.
        """
        if preserve_topology:
            op = self.impl['topology_preserve_simplify']
        else:
            op = self.impl['simplify']
        return geom_factory(op(self, tolerance))

    # Binary operations
    # -----------------

    def difference(self, other):
        """Returns the difference of the geometries"""
        return geom_factory(self.impl['difference'](self, other))
    
    def intersection(self, other):
        """Returns the intersection of the geometries"""
        return geom_factory(self.impl['intersection'](self, other))

    def symmetric_difference(self, other):
        """Returns the symmetric difference of the geometries 
        (Shapely geometry)"""
        return geom_factory(self.impl['symmetric_difference'](self, other))

    def union(self, other):
        """Returns the union of the geometries (Shapely geometry)"""
        return geom_factory(self.impl['union'](self, other))

    # Unary predicates
    # ----------------

    @property
    def has_z(self):
        """True if the geometry's coordinate sequence(s) have z values (are
        3-dimensional)"""
        return bool(self.impl['has_z'](self))

    @property
    def is_empty(self):
        """True if the set of points in this geometry is empty, else False"""
        return bool(self.impl['is_empty'](self)) or (self._geom is None)

    @property
    def is_ring(self):
        """True if the geometry is a closed ring, else False"""
        return bool(self.impl['is_ring'](self))

    @property
    def is_simple(self):
        """True if the geometry is simple, meaning that any self-intersections 
        are only at boundary points, else False"""
        return bool(self.impl['is_simple'](self))

    @property
    def is_valid(self):
        """True if the geometry is valid (definition depends on sub-class), 
        else False"""
        return bool(self.impl['is_valid'](self))

    # Binary predicates
    # -----------------

    def relate(self, other):
        """Returns the DE-9IM intersection matrix for the two geometries 
        (string)"""
        return self.impl['relate'](self, other)

    def contains(self, other):
        """Returns True if the geometry contains the other, else False"""
        return bool(self.impl['contains'](self, other))

    def crosses(self, other):
        """Returns True if the geometries cross, else False"""
        return bool(self.impl['crosses'](self, other))

    def disjoint(self, other):
        """Returns True if geometries are disjoint, else False"""
        return bool(self.impl['disjoint'](self, other))

    def equals(self, other):
        """Returns True if geometries are equal, else False"""
        return bool(self.impl['equals'](self, other))

    def intersects(self, other):
        """Returns True if geometries intersect, else False"""
        return bool(self.impl['intersects'](self, other))

    def overlaps(self, other):
        """Returns True if geometries overlap, else False"""
        return bool(self.impl['overlaps'](self, other))

    def touches(self, other):
        """Returns True if geometries touch, else False"""
        return bool(self.impl['touches'](self, other))

    def within(self, other):
        """Returns True if geometry is within the other, else False"""
        return bool(self.impl['within'](self, other))

    def equals_exact(self, other, tolerance):
        """Returns True if geometries are equal to within a specified 
        tolerance"""
        # return BinaryPredicateOp('equals_exact', self)(other, tolerance)
        return bool(self.impl['equals_exact'](self, other, tolerance))

    def almost_equals(self, other, decimal=6):
        """Returns True if geometries are equal at all coordinates to a 
        specified decimal place"""
        return self.equals_exact(other, 0.5 * 10**(-decimal))

    # Linear referencing
    # ------------------

    @delegated
    def project(self, other, normalized=False):
        """Returns the distance along this geometry to a point nearest the 
        specified point
        
        If the normalized arg is True, return the distance normalized to the
        length of the linear geometry.
        """ 
        if normalized:
            op = self.impl['project_normalized']
        else:
            op = self.impl['project']
        return op(self, other)
           
    @delegated
    def interpolate(self, distance, normalized=False):
        """Return a point at the specified distance along a linear geometry
        
        If the normalized arg is True, the distance will be interpreted as a
        fraction of the geometry's length.
        """
        if normalized:
            op = self.impl['interpolate_normalized']
        else:
            op = self.impl['interpolate']
        return geom_factory(op(self, distance))


class BaseMultipartGeometry(BaseGeometry):

    def shape_factory(self, *args):
        # Factory for part instances, usually a geometry class
        raise NotImplementedError("To be implemented by derived classes")

    @property
    def ctypes(self):
        raise NotImplementedError(
        "Multi-part geometries have no ctypes representations")

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        raise NotImplementedError(
        "Multi-part geometries do not themselves provide the array interface")

    def _get_coords(self):
        raise NotImplementedError(
        "Sub-geometries may have coordinate sequences, but collections do not")

    def _set_coords(self, ob):
        raise NotImplementedError(
        "Sub-geometries may have coordinate sequences, but collections do not")

    @property
    def coords(self):
        raise NotImplementedError(
        "Multi-part geometries do not provide a coordinate sequence")

    @property
    def geoms(self):
        if self.is_empty:
            return []
        return GeometrySequence(self, self.shape_factory)

    def __iter__(self):
        if not self.is_empty:
            return iter(self.geoms)
        else:
            return iter([])

    def __len__(self):
        if not self.is_empty:
            return len(self.geoms)
        else:
            return 0

    def __getitem__(self, index):
        if not self.is_empty:
            return self.geoms[index]
        else:
            return ()[index]


class GeometrySequence(object):
    """
    Iterative access to members of a homogeneous multipart geometry.
    """

    # Attributes
    # ----------
    # _factory : callable
    #     Returns instances of Shapely geometries
    # _geom : c_void_p
    #     Ctypes pointer to the parent's GEOS geometry
    # _ndim : int
    #     Number of dimensions (2 or 3, generally)
    # __p__ : object
    #     Parent (Shapely) geometry
    shape_factory = None
    _geom = None
    __p__ = None
    _ndim = None

    def __init__(self, parent, type):
        self.shape_factory = type
        self.__p__ = parent

    def _update(self):
        self._geom = self.__p__._geom
        self._ndim = self.__p__._ndim
        
    def _get_geom_item(self, i):
        g = self.shape_factory()
        g._owned = True
        g._geom = lgeos.GEOSGetGeometryN(self._geom, i)
        g._ndim = self._ndim
        g.__p__ = self
        return g

    def __iter__(self):
        self._update()
        for i in xrange(self.__len__()):
            yield self._get_geom_item(i)

    def __len__(self):
        self._update()
        return lgeos.GEOSGetNumGeometries(self._geom)

    def __getitem__(self, key):
        self._update()
        m = self.__len__()
        if isinstance(key, int):
            if key + m < 0 or key >= m:
                raise IndexError("index out of range")
            if key < 0:
                i = m + key
            else:
                i = key
            return self._get_geom_item(i)
        elif isinstance(key, slice):
            if type(self) == HeterogeneousGeometrySequence:
                raise TypeError(
                    "Heterogenous geometry collections are not sliceable")
            res = []
            start, stop, stride = key.indices(m)
            for i in xrange(start, stop, stride):
                res.append(self._get_geom_item(i))
            return type(self.__p__)(res or None)
        else:
            raise TypeError("key must be an index or slice")

    @property
    def _longest(self):
        max = 0
        for g in iter(self):
            l = len(g.coords)
            if l > max:
                max = l


class HeterogeneousGeometrySequence(GeometrySequence):
    """
    Iterative access to a heterogeneous sequence of geometries.
    """

    def __init__(self, parent):
        super(HeterogeneousGeometrySequence, self).__init__(parent, None)

    def _get_geom_item(self, i):
        sub = lgeos.GEOSGetGeometryN(self._geom, i)
        g = geom_factory(sub)
        g._owned = True
        return g

# Test runner
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
