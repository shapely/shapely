"""Load/dump geometries using the well-known text (WKT) format
"""

from shapely.geos import lgeos

# Pickle-like convenience functions

def loads(data):
    """Load a geometry from a WKT string."""
    return lgeos.wkt_reader.read(data)

def load(fp):
    """Load a geometry from an open file."""
    data = fp.read()
    return loads(data)

def dumps(ob):
    """Dump a WKT representation of a geometry to a string."""
    return lgeos.wkt_writer.write(ob)

def dump(ob, fp):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob))
