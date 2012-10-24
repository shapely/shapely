from cPickle import dumps, loads, HIGHEST_PROTOCOL
import unittest
from shapely import geometry

class TwoDeeTestCase(unittest.TestCase):
    """."""
    def test_linestring(self):
        l = geometry.LineString(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
        self.failUnlessEqual(l._ndim, 2)
        s = dumps(l, HIGHEST_PROTOCOL)
        t = loads(s)
        self.failUnlessEqual(t._ndim, 2)

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(TwoDeeTestCase)
