"""
Load/dump geometries using JSON.

Includes a JSON-based abstract geometry factory.
"""

import geojson

def json_factory(ob):
    """Attempts to create a geometry from JSON.
    
    Else, returns the original data.
    """
    try:
        coords = ob.get('coordinates', [])
        geom_type = str(ob.get('type'))
        mod = __import__(
            'shapely.geometry', 
            globals(), 
            locals(), 
            [geom_type],
            )
        geom_class = getattr(mod, geom_type)
        return geom_class(*coords)
    except (KeyError, TypeError):
        raise 
    return ob

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

