import unittest
from shapely.geometry.polygon import LinearRing, orient, Polygon

class RingOrientationTestCase(unittest.TestCase):
    def test_ccw(self):
        ring = LinearRing([(1,0),(0,1),(0,0)])
        self.failUnless(ring.is_ccw)
    def test_cw(self):
        ring = LinearRing([(0,0),(0,1),(1,0)])
        self.failIf(ring.is_ccw)

class PolygonOrienterTestCase(unittest.TestCase):
    def test_no_holes(self):
        ring = LinearRing([(0,0),(0,1),(1,0)])
        polygon = Polygon(ring)
        self.failIf(polygon.exterior.is_ccw)
        polygon = orient(polygon, 1)
        self.failUnless(polygon.exterior.is_ccw)
    def test_holes(self):
        polygon = Polygon([(0,0),(0,1),(1,0)], 
                        [[(0.5,0.25),(0.25,0.5),(0.25,0.25)]])
        self.failIf(polygon.exterior.is_ccw)
        self.failUnless(polygon.interiors[0].is_ccw)
        polygon = orient(polygon, 1)
        self.failUnless(polygon.exterior.is_ccw)
        self.failIf(polygon.interiors[0].is_ccw)

def test_suite():
    loader = unittest.TestLoader()
    return unittest.TestSuite([
        loader.loadTestsFromTestCase(RingOrientationTestCase),
        loader.loadTestsFromTestCase(PolygonOrienterTestCase)])

