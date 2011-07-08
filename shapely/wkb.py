"""Load/dump geometries using the well-known binary (WKB) format
"""

from ctypes import pointer, c_size_t, c_char_p, c_void_p

from shapely.geos import lgeos, ReadingError


# Pickle-like convenience functions

def deserialize(data):
    geom = lgeos.GEOSGeomFromWKB_buf(c_char_p(data), c_size_t(len(data)));
    if not geom:
        raise ReadingError(
            "Could not create geometry because of errors while reading input.")
    return geom

def loads(data):
    """Load a geometry from a WKB string."""
    from shapely.geometry.base import geom_factory
    return geom_factory(deserialize(data))

def load(fp):
    """Load a geometry from an open file."""
    data = fp.read()
    return loads(data)

def dumps(ob):
    """Dump a WKB representation of a geometry to a byte string."""
    if ob is None or ob._geom is None:
        raise ValueError("Null geometry supports no operations")
    size = c_size_t()
    return lgeos.GEOSGeomToWKB_buf(c_void_p(ob._geom), pointer(size))

def dump(ob, fp):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob))

