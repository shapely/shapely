"""Proxy for coordinates stored outside Shapely geometries
"""

from shapely.geos import lgeos
from shapely import wkb

EMPTY = wkb.deserialize('010700000000000000'.decode('hex'))


class CachingGeometryProxy(object):

    context = None
    factory = None
    __geom__ = EMPTY
    _gtag = None

    def __init__(self, context):
        self.context = context

    @property
    def _is_empty(self):
        return self.__geom__ in [EMPTY, None]

    def empty(self):
        if not self._is_empty:
            lgeos.GEOSGeom_destroy(self.__geom__)
        self.__geom__ = EMPTY

    @property
    def _geom(self):
        """Keeps the GEOS geometry in synch with the context."""
        gtag = self.gtag()
        if gtag != self._gtag or self._is_empty:
            self.empty()
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
        if gtag != self._gtag or self._is_empty:
            self.empty()
            self.__geom__, n = self.factory(self.context[0], self.context[1])
        self._gtag = gtag
        return self.__geom__
