"""
Load/dump geometries using the well-known text (WKT) format.
"""

from ctypes import byref, c_int, c_size_t, c_char_p, string_at

from shapely.geos import lgeos, free, allocated_c_char_p, ReadingError
from shapely.geometry.base import geom_factory


# Pickle-like convenience functions

def loads(data):
    """Load a geometry from a WKT string."""
    geom = lgeos.GEOSGeomFromWKT(c_char_p(data))
    if not geom:
        raise ReadingError, \
        "Could not create geometry because of errors while reading input."
    return geom_factory(geom)

def load(fp):
    """Load a geometry from an open file."""
    data = fp.read()
    return loads(data)

def dumps(ob):
    """Dump a WKB representation of a geometry to a byte string."""
    func = lgeos.GEOSGeomToWKT
    def errcheck(result, func, argtuple):
        retval = result.value
        free(result)
        return retval
    func.restype = allocated_c_char_p
    func.errcheck = errcheck
    return func(ob._geom)

def dump(ob, fp):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob))

