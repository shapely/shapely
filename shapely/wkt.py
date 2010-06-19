"""Load/dump geometries using the well-known text (WKT) format
"""

from ctypes import c_char_p

from shapely.geos import lgeos, ReadingError


# Pickle-like convenience functions

def loads(data):
    """Load a geometry from a WKT string."""
    from shapely.geometry.base import geom_factory
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
    if ob is None or ob._geom is None:
        raise ValueError("Null geometry supports no operations")
    return lgeos.GEOSGeomToWKT(ob._geom)

def dump(ob, fp):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob))

