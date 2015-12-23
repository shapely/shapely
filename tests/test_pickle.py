from . import unittest
from shapely import geometry

import sys
if sys.version_info[0] >= 3:
    from pickle import dumps, loads, HIGHEST_PROTOCOL
else:
    from cPickle import dumps, loads, HIGHEST_PROTOCOL


class TwoDeeTestCase(unittest.TestCase):

    def test_linestring(self):
        l = geometry.LineString(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
        self.assertEqual(l._ndim, 2)
        s = dumps(l, HIGHEST_PROTOCOL)
        t = loads(s)
        self.assertEqual(t._ndim, 2)


GEOMETRY_TYPES = [
    geometry.Point,
    geometry.LineString,
    geometry.Polygon,
    geometry.LinearRing,
    geometry.MultiPoint,
    geometry.MultiLineString,
    geometry.MultiPolygon,
    geometry.GeometryCollection,
]
class RoundTripTestCase(unittest.TestCase):
    def test_roundtrip_empty(self):
        for GEOMETRY_TYPE in GEOMETRY_TYPES:
            geom = GEOMETRY_TYPE()
            data = dumps(geom)
            result = loads(data)
            self.assertEqual(geom.wkt, result.wkt)
            self.assertEqual(geom, result)

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(TwoDeeTestCase)
