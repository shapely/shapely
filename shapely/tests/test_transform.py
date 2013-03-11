import unittest
from shapely import geometry
from shapely.ops import transform

class IdentityTestCase(unittest.TestCase):
    """New geometry/coordseq method 'xy' makes numpy interop easier"""
    def func(self, x, y, z=None):
        return tuple(filter(None, [x, y, z]))
    def test_point(self):
        g = geometry.Point(0, 1)
        h = transform(self.func, g)
        self.failUnlessEqual(h.geom_type, 'Point')
        self.failUnlessEqual(list(h.coords), [(0, 1)])
    def test_line(self):
        g = geometry.LineString(((0, 1), (2, 3)))
        h = transform(self.func, g)
        self.failUnlessEqual(h.geom_type, 'LineString')
        self.failUnlessEqual(list(h.coords), [(0, 1), (2, 3)])
    def test_polygon(self):
        g = geometry.Point(0, 1).buffer(1.0)
        h = transform(self.func, g)
        self.failUnlessEqual(h.geom_type, 'Polygon')
        self.failUnlessAlmostEqual(g.area, h.area)
    def test_multipolygon(self):
        g = geometry.MultiPoint([(0, 1), (0, 4)]).buffer(1.0)
        h = transform(self.func, g)
        self.failUnlessEqual(h.geom_type, 'MultiPolygon')
        self.failUnlessAlmostEqual(g.area, h.area)

class LambdaTestCase(unittest.TestCase):
    """New geometry/coordseq method 'xy' makes numpy interop easier"""
    def test_point(self):
        g = geometry.Point(0, 1)
        h = transform(lambda x, y, z=None: (x+1.0, y+1.0), g)
        self.failUnlessEqual(h.geom_type, 'Point')
        self.failUnlessEqual(list(h.coords), [(1.0, 2.0)])
    def test_line(self):
        g = geometry.LineString(((0, 1), (2, 3)))
        h = transform(lambda x, y, z=None: (x+1.0, y+1.0), g)
        self.failUnlessEqual(h.geom_type, 'LineString')
        self.failUnlessEqual(list(h.coords), [(1.0, 2.0), (3.0, 4.0)])
    def test_polygon(self):
        g = geometry.Point(0, 1).buffer(1.0)
        h = transform(lambda x, y, z=None: (x+1.0, y+1.0), g)
        self.failUnlessEqual(h.geom_type, 'Polygon')
        self.failUnlessAlmostEqual(g.area, h.area)
        self.failUnlessAlmostEqual(h.centroid.x, 1.0)
        self.failUnlessAlmostEqual(h.centroid.y, 2.0)
    def test_multipolygon(self):
        g = geometry.MultiPoint([(0, 1), (0, 4)]).buffer(1.0)
        h = transform(lambda x, y, z=None: (x+1.0, y+1.0), g)
        self.failUnlessEqual(h.geom_type, 'MultiPolygon')
        self.failUnlessAlmostEqual(g.area, h.area)
        self.failUnlessAlmostEqual(h.centroid.x, 1.0)
        self.failUnlessAlmostEqual(h.centroid.y, 3.5)

def test_suite():
    loader = unittest.TestLoader()
    return unittest.TestSuite([
        unittest.TestLoader().loadTestsFromTestCase(IdentityTestCase),
        unittest.TestLoader().loadTestsFromTestCase(LambdaTestCase)])
