"""Load/dump geometries using the well-known binary (WKB) format
"""

from shapely import geos

# Pickle-like convenience functions

def loads(data, hex=False):
    """Load a geometry from a WKB byte string, or hex-encoded string if
    ``hex=True``.
    """
    reader = geos.WKBReader(geos.lgeos)
    if hex:
        return reader.read_hex(data)
    else:
        return reader.read(data)

def load(fp, hex=False):
    """Load a geometry from an open file."""
    data = fp.read()
    return loads(data, hex=hex)

def dumps(ob, hex=False, **kw):
    """Dump a WKB representation of a geometry to a byte string, or a
    hex-encoded string if ``hex=True``.

    See available keyword output settings in ``shapely.geos.WKBWriter``."""
    writer = geos.WKBWriter(geos.lgeos, **kw)
    if hex:
        return writer.write_hex(ob)
    else:
        return writer.write(ob)


def dump(ob, fp, hex=False, **kw):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob, hex=hex, **kw))
