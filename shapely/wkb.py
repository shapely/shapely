"""
Load/dump geometries using the well-known binary (WKB) format.
"""

from ctypes import byref, c_int, c_size_t, c_char_p, string_at
from ctypes import c_void_p, c_size_t

from shapely.geos import lgeos, free, ReadingError
from shapely.geometry.base import geom_factory


# Pickle-like convenience functions

def loads(data):
    """Load a geometry from a WKB string."""
    geom = lgeos.GEOSGeomFromWKB_buf(c_char_p(data), c_size_t(len(data)));
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
    func = lgeos.GEOSGeomToWKB_buf
    size = c_size_t()
    def errcheck(result, func, argtuple):
        if not result: return None
        retval = string_at(result, size.value)[:]
        free(result)
        return retval
    func.errcheck = errcheck
    return func(c_void_p(ob._geom), byref(size))

def dump(ob, fp):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob))

