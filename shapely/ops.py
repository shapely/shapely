"""
Support for various GEOS geometry operations
"""

from ctypes import byref, c_void_p

from shapely.geos import lgeos
from shapely.geometry.base import geom_factory, BaseGeometry
from shapely.geometry import asShape, asLineString, asMultiLineString

def shapeup(ob):
    if isinstance(ob, BaseGeometry):
        return ob
    else:
        try:
            return asShape(ob)
        except ValueError:
            return asLineString(ob)

def polygonize(iterator):
    """Creates polygons from a list of LineString objects.
    """
    lines = [shapeup(ob) for ob in iterator]
    geom_array_type = c_void_p * len(lines)
    geom_array = geom_array_type()
    for i, line in enumerate(lines):
        geom_array[i] = line._geom
    product = lgeos.GEOSPolygonize(byref(geom_array), len(lines))
    collection = geom_factory(product)
    for g in collection.geoms:
        clone = lgeos.GEOSGeom_clone(g._geom)
        g = geom_factory(clone)
        g._owned = False
        yield g

def linemerge(lines): 
    """Merges all connected lines. Returns a LineString or MultiLineString 
    when lines are not contiguous. 
        
    Parameters 
    ---------- 
    lines : MultiLineString, sequence of lines, or sequence of coordinates
    """ 
    multilinestring = None 
    if hasattr(lines, 'type') and lines.type == 'MultiLineString': 
        multilinestring = lines 
    elif hasattr(lines, '__iter__'): 
        try: 
            multilinestring = asMultiLineString([ls.coords for ls in lines]) 
        except AttributeError: 
            multilinestring = asMultiLineString(lines) 
    if multilinestring is None: 
        raise ValueError("Cannot linemerge %s" % lines)
    result = lgeos.GEOSLineMerge(multilinestring._geom) 
    return geom_factory(result)   

def cascaded_union(geoms):
    """Returns the union of a sequence of geometries
    
    This is the most efficient method of dissolving many polygons.
    """
    L = len(geoms)
    subs = (c_void_p * L)()
    for i, g in enumerate(geoms):
        subs[i] = g._geom
    collection = lgeos.GEOSGeom_createCollection(6, subs, L)
    return geom_factory(lgeos.GEOSUnionCascaded(collection))

