"""Load/dump geometries using the well-known text (WKT) format
"""

from shapely.geometry.base import geom_from_wkt, geom_to_wkt

# Pickle-like convenience functions

def loads(data):
    """Load a geometry from a WKT string."""
    return geom_from_wkt(data) #factory(geom)

def load(fp):
    """Load a geometry from an open file."""
    data = fp.read()
    return loads(data)

def dumps(ob):
    """Dump a WKB representation of a geometry to a byte string."""
    return geom_to_wkt(ob)

def dump(ob, fp):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob))

