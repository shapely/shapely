from shapely.geos import lgeos
from shapely.geometry.base import geom_factory, BaseGeometry
from shapely.geometry import asShape, asLineString
from ctypes import byref, c_void_p

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
