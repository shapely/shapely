"""Load/dump geometries using the well-known text (WKT) format
"""

from shapely import geos

# Pickle-like convenience functions

def loads(data):
    """Load a geometry from a WKT string."""
    return geos.WKTReader(geos.lgeos).read(data)

def load(fp):
    """Load a geometry from an open file."""
    data = fp.read()
    return loads(data)

def dumps(ob, trim=False, **kw):
    """Dump a WKT representation of a geometry to a string.

    See available keyword output settings in ``shapely.geos.WKTWriter``.
    """
    return geos.WKTWriter(geos.lgeos, trim=trim, **kw).write(ob)

def dump(ob, fp, **settings):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob, **settings))
