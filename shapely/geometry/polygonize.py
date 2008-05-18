from shapely.geos import lgeos
from shapely.geometry.base import geom_factory
from ctypes import byref, c_void_p, c_double

def polygonize(lines):
    """Creates polygons from a list of LineString objects.
    """
    geom_array_type = c_void_p * len(lines)
    geom_array = geom_array_type()
    for i, line in enumerate(lines):
        geom_array[i] = line._geom
    product = lgeos.GEOSPolygonize(byref(geom_array), len(lines))
    return geom_factory(product)
