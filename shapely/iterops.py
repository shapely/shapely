"""
Iterative forms of operations
"""
from warnings import warn
from ctypes import c_char_p, c_size_t
from shapely.geos import lgeos, PredicateError


def geos_from_geometry(geom):
    warn("`geos_from_geometry` is deprecated. Use geometry's `wkb` property "
         "instead.", DeprecationWarning)
    data = geom.to_wkb()
    return lgeos.GEOSGeomFromWKB_buf(
                        c_char_p(data),
                        c_size_t(len(data))
                        )


class IterOp(object):
    
    """A generating non-data descriptor.
    """
    
    def __init__(self, fn):
        self.fn = fn
    
    def __call__(self, context, iterator, value=True):
        if context._geom is None:
            raise ValueError("Null geometry supports no operations")
        for item in iterator:
            try:
                this_geom, ob = item
            except TypeError:
                this_geom = item
                ob = this_geom
            if not this_geom._geom:
                raise ValueError("Null geometry supports no operations")
            retval = self.fn(context._geom, this_geom._geom)
            if retval == 2:
                raise PredicateError(
                    "Failed to evaluate %s" % repr(self.fn))
            elif bool(retval) == value:
                yield ob


# utilities
disjoint = IterOp(lgeos.GEOSDisjoint)
touches = IterOp(lgeos.GEOSTouches)
intersects = IterOp(lgeos.GEOSIntersects)
crosses = IterOp(lgeos.GEOSCrosses)
within = IterOp(lgeos.GEOSWithin)
contains = IterOp(lgeos.GEOSContains)
overlaps = IterOp(lgeos.GEOSOverlaps)
equals = IterOp(lgeos.GEOSEquals)

