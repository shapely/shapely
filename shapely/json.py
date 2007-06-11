#

import geojson

from shapely.factory import json_factory


# Pickle-like convenience functions

def loads(data):
    """Load a geometry from a GeoJSON representation."""
    return geojson.loads(data, json_factory)

def load(fp):
    """Load a geometry from an open file."""
    data = fp.read()
    return loads(data)

def dumps(ob):
    """Dump a GeoJSON representation of a geometry to a string."""
    return geojson.dumps(ob)

def dump(ob, fp):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob))

