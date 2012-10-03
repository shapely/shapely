"""Load/dump geometries using the well-known binary (WKB) format
"""

from shapely.geometry.base import geom_from_wkb, geom_to_wkb

# Pickle-like convenience functions

def loads(data):
    """Load a geometry from a WKB string."""
    return geom_from_wkb(data)

def load(fp):
    """Load a geometry from an open file."""
    data = fp.read()
    return loads(data)

def dumps(ob):
    """Dump a WKB representation of a geometry to a byte string."""
    return geom_to_wkb(ob)

def dump(ob, fp):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob))

