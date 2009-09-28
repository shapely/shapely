from shapely.geos import lgeos
from shapely.predicates import BinaryPredicate

class PreparedGeometry(object):
    """
    A geometry prepared for efficient comparison to a set of other geometries.
    
    Example:
      
      >>> from shapely.geometry import Point, Polygon
      >>> triangle = Polygon(((0.0, 0.0), (1.0, 1.0), (1.0, -1.0)))
      >>> p = prep(triangle)
      >>> p.intersects(Point(0.5, 0.5))
      True
    
    """
    
    __geom__ = None
    
    def __init__(self, context):
        self.context = context
        self.__geom__ = lgeos.GEOSPrepare(self.context._geom)
    
    def __del__(self):
        if self.__geom__ is not None:
            lgeos.GEOSPreparedGeom_destroy(self.__geom__)
        self.__geom__ = None
        self.context = None
        
    @property
    def _geom(self):
        return self.__geom__
        
    intersects = BinaryPredicate(lgeos.GEOSPreparedIntersects)
    contains = BinaryPredicate(lgeos.GEOSPreparedContains)
    contains_properly = BinaryPredicate(lgeos.GEOSPreparedContainsProperly)
    covers = BinaryPredicate(lgeos.GEOSPreparedCovers)

def prep(geom):
    return PreparedGeometry(geom)