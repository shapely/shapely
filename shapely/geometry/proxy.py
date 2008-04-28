"""
Proxy for coordinates stored outside Shapely geometries.
"""

from shapely.geos import lgeos


class CachingGeometryProxy(object):

    context = None
    factory = None
    __geom__ = None
    _gtag = None

    def __init__(self, context):
        self.context = context

    @property
    def _geom(self):
        """Keeps the GEOS geometry in synch with the context."""
        gtag = self.gtag()
        if gtag != self._gtag:
            if self.__geom__ is not None:
                lgeos.GEOSGeom_destroy(self.__geom__)
            self.__geom__, n = self.factory(self.context)
        elif self.__geom__ is None:
            self.__geom__, n = self.factory(self.context)
        self._gtag = gtag
        return self.__geom__
        
    def gtag(self):
        return hash(repr(self.context))


class PolygonProxy(CachingGeometryProxy):

    @property
    def _geom(self):
        """Keeps the GEOS geometry in synch with the context."""
        gtag = self.gtag()
        if gtag != self._gtag:
            if self.__geom__ is not None:
                lgeos.GEOSGeom_destroy(self.__geom__)
            self.__geom__, n = self.factory(self.context[0], self.context[1])
        elif self.__geom__ is None:
            self.__geom__, n = self.factory(self.context[0], self.context[1])
        self._gtag = gtag
        return self.__geom__
