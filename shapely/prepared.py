"""
Support for GEOS prepared geometry operations.
"""

from shapely.geos import lgeos
from shapely.impl import DefaultImplementation, delegated


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
   
    impl = DefaultImplementation
    
    def __init__(self, context):
        self.context = context
        self.__geom__ = lgeos.GEOSPrepare(self.context._geom)
    
    def __del__(self):
        if self.__geom__ is not None:
            try:
                lgeos.GEOSPreparedGeom_destroy(self.__geom__)
            except AttributeError:
                pass # lgeos might be empty on shutdown
        self.__geom__ = None
        self.context = None
    
    @property
    def _geom(self):
        return self.__geom__
   
    @delegated
    def intersects(self, other):
        return bool(self.impl['prepared_intersects'](self, other))

    @delegated
    def contains(self, other):
        return bool(self.impl['prepared_contains'](self, other))

    @delegated
    def contains_properly(self, other):
        return bool(self.impl['prepared_contains_properly'](self, other))

    @delegated
    def covers(self, other):
        return bool(self.impl['prepared_covers'](self, other))


def prep(ob):
    """Creates and returns a prepared geometric object."""
    return PreparedGeometry(ob)

