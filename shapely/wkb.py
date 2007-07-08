"""
Load/dump geometries using the well-known binary (WKB) format.
"""

from ctypes import byref, c_int, c_size_t, c_char_p, string_at

from shapely.geos import lgeos, ReadingError
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
    size = c_int()
    bytes = lgeos.GEOSGeomToWKB_buf(ob._geom, byref(size))
    return string_at(bytes, size.value)

def dump(ob, fp):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob))

